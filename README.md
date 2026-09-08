# Weather AI Home Assistant Add-on

## CI/CD & self-deployment setup

This repo *is* a Home Assistant add-on repository — `config.yaml` sits at the repo
root. Home Assistant's own Supervisor is the deployment mechanism: it polls this
repo, and when it sees the `version` in `config.yaml` bumped, it rebuilds and
(if auto-update is on) installs the new version by itself. GitHub Actions
(`.github/workflows/ci.yml`) only *validates* that every push/PR still builds
cleanly for `amd64`/`aarch64` — it doesn't push or deploy anything.

One-time setup:

1. **Replace placeholders.** Search this repo for `OWNER/REPO` (in `build.yaml`
   and `repository.yaml`) and swap in your actual GitHub `owner/repo`.
2. **Push this repo to GitHub.**
3. **Upload the large model files as a Release**, since they're too big (up to
   ~490MB) to commit as normal git blobs and are `.gitignore`d:
   - GitHub → your repo → **Releases → Draft a new release**
   - Tag: `models` (this exact tag — `build.yaml`'s `MODEL_BASE_URL` points at it)
   - Attach these 4 files from this folder:
     `weather_rf_model_5m.joblib`, `weather_rf_model_10m.joblib`,
     `weather_rf_model_30m.joblib`, `weather_temp_model_30m.joblib`
   - Publish. When you retrain and get new versions of these files, edit the
     `models` release and re-upload (overwrite) the assets — no code change needed.
4. **Add the repo to Home Assistant:** Settings → Add-ons → Add-on Store →
   ⋮ (top right) → Repositories → paste your GitHub repo URL → Add.
5. Install the "Weather AI" add-on from the store, then toggle **Auto update**
   for it (Add-on page → ⋮ → three-dot menu). This is what makes future
   `version` bumps in `config.yaml` install themselves without you touching HA.

To ship a new version: bump `version` in `config.yaml`, commit, push to `main`.
CI validates the build; Supervisor picks up the bump and (auto-)updates the addon.

---

FastAPI service for receiving weather sensor readings, saving them to daily CSV files
and (optionally) a MariaDB database, returning 5m/10m/30m/60m/120m rain predictions from the
trained model, and pushing rain-alert notifications to Telegram when the rain state
changes.

## Home Assistant entities

On every `/reading` POST, the add-on pushes state updates straight into Home Assistant
via the Supervisor's Core API proxy (`homeassistant_api: true` in `config.yaml` — no
MQTT broker needed):

- `sensor.weather_ai_temperature`, `sensor.weather_ai_humidity`, `sensor.weather_ai_pressure`
- `sensor.weather_ai_last_update` — timestamp heartbeat ของ prediction ล่าสุด
  (อัปเดตทุกครั้งที่คำนวณสำเร็จ แม้ค่าเซนเซอร์ที่ปัดเศษแล้วจะไม่เปลี่ยน)
- `sensor.weather_ai_heat_index` (attribute: `comfort_level`)
- `sensor.weather_ai_dew_point` — from the `dew_point` feature already computed in
  `weather_features_lib.py` (no new modeling, just exposed)
- `sensor.weather_ai_pressure_trend` — hPa change over the last hour (`pressure_trend_60m`
  feature), unit `hPa/h`
- `sensor.weather_ai_trend_arrow` — text state (`ขึ้น ↑` / `ลง ↓` / `คงที่ →`), derived from
  `pressure_trend` with a ±0.5 hPa/h steady band (see `PRESSURE_TREND_STEADY_BAND` in
  `predict_weather_ai.py`)
- `sensor.weather_ai_rain_probability_5m` / `_10m` / `_30m` (%, attributes: `rain_alert`, `threshold`)
- `binary_sensor.weather_ai_raining_now`
- `binary_sensor.weather_ai_rain_alert` (attributes: `next_rain_alert_horizon`, `alert_message`)
- `sensor.weather_ai_tmd_warning_count` / `binary_sensor.weather_ai_tmd_warning` —
  area-matched official TMD CAP warnings (source/severity/expiry attributes)
- `sensor.weather_ai_rain_stop_eta` — experimental ETA and p20–p80 remaining-minute
  interval; becomes `unavailable` when history is stale or insufficient

A ready-made dashboard is in `dashboard.yaml`. It uses the long-supported `masonry`
view format (with a top-level `views:` wrapper), so it also works on Home Assistant
versions that do not support the newer `sections` view type. The compatibility version
uses only the core entities that every Weather AI prediction publishes (rain status,
5/10/30-minute probabilities, station values and history); optional radar/cloud/TMD
entities are deliberately omitted so an older HA instance can save the YAML even before
those entities have been created. It uses only stock HA cards, no HACS required.

To install: **Settings → Dashboards → + Add Dashboard → New dashboard from scratch**, then
open the new dashboard's three-dot menu → **Edit Dashboard → three-dot menu → Raw configuration
editor**, and paste in the contents of `dashboard.yaml`.

If you already have a large dashboard with an existing `views:` list, do not paste the
full file inside another `views:` key. Replace or append one view item using
`dashboard_view.yaml` (it starts with `- title: Weather AI`) under the existing `views:` list.
For the full Weather AI view, including the optional radar/cloud/TMD/rain-stop cards,
use `dashboard_view_complete.yaml` the same way. Those optional entities appear after
the corresponding add-on data source has produced its first result; until then HA may
show them as unavailable.

## Configuration options

- `db_host`, `db_port`, `db_user`, `db_password`, `db_name` — MariaDB connection.
  Leave blank to skip DB persistence (CSV storage still works either way).
- `telegram_bot_token`, `telegram_chat_id` — Telegram Bot API credentials. Leave
  blank to disable Telegram notifications.

Rain alerts are sent to Telegram only on state changes (rain starts/stops, a new
rain-alert horizon triggers, or an alert clears) — not on every reading — to avoid
spamming the chat every minute. `alert_cooldown_seconds` adds a durable per-event
cooldown across restarts. To avoid a rain sensor's brief dry flicker ending an active
shower, `rain_stop_confirm_polls` requires consecutive dry source polls before sending
the stop message (default 3 polls, about 3 minutes at the default 60-second HA poll).
A dry spell that is shorter than this stays within the same rain event; after a confirmed
stop, a real return sends the distinct **rain resumed** event, so it is not suppressed by
the previous rain-start cooldown. Rain-start and model-alert messages include the
multi-horizon probabilities, pressure trend, model version, feature-history coverage,
sensor-data age, and clearly labelled experimental radar/cloud context when available.
Use `/status` in Telegram for the same source/model health on demand; `/forecast`
includes the quality metadata alongside the nowcast. NWP bias correction is opt-in: set
`nwp_bias_correction_path` to a reviewed JSON artifact; the API keeps both raw
and corrected hourly temperatures. For calibrated temperature ranges, set
`nwp_temp_interval_path` to an artifact produced by `train_temperature_intervals.py`
from a chronological holdout; the API leaves the interval null when no compatible
artifact is configured. `/forecast` also exposes an experimental rain-stop estimate
when at least three completed local rain events are available, and reads active
area-matched warnings from the TMD CAP feed with source/expiry metadata.
Revision messages can be sent proactively only after setting
`revision_alert_enabled` to `true`; the default is off and the cooldown/poll
interval are controlled by `revision_alert_cooldown_seconds` and
`revision_poll_seconds`. Telegram command polling recreates its HTTP session after
a connection reset, backs off from 5 to 60 seconds, and honors Telegram's `429`
`retry_after` value.

### ดึงเซนเซอร์จาก Home Assistant โดยตรง (14.6)

โหมดนี้ปิดไว้เป็นค่าเริ่มต้นเพื่อป้องกันการอ่าน entity ผิดตัว เปิดใช้โดยตั้ง
`ha_source_enabled: true` แล้วระบุ entity ต้นทางอย่างน้อย 3 ตัว:
`ha_temp_entity`, `ha_humidity_entity`, `ha_pressure_entity` (เช่น
`sensor.outdoor_temperature`). ระบุ `ha_rain_entity` และ `ha_light_entity` ได้ถ้ามี
เซนเซอร์เหล่านั้นด้วย; ห้ามใช้ entity ที่ขึ้นต้นด้วย `sensor.weather_ai_` หรือ
`binary_sensor.weather_ai_` เพราะเป็น output ของ add-on เองและจะทำให้เกิด feedback loop.

Supervisor จะส่ง `SUPERVISOR_TOKEN` ให้อัตโนมัติเมื่อ `homeassistant_api: true` และ
add-on จะอ่านแบบ read-only จาก `/states/<entity_id>` ทุก `ha_poll_seconds` วินาที
(ค่าเริ่มต้น 60) ตรวจเวลาของข้อมูลไม่ให้เกิน `ha_stale_after_seconds` (180 วินาที),
แปลง °F/K เป็น °C และ Pa/kPa/inHg เป็น hPa, พร้อมตัดค่า `unknown`/`unavailable`.
สำหรับ `binary_sensor` ฝน ค่า `on/off` ที่คงเดิมถือเป็น state ที่ใช้งานได้แม้
`last_updated` เก่า (เพราะ binary sensor ไม่ได้ส่ง heartbeat ทุกนาที); แต่ถ้าเป็น
`unknown`/`unavailable` หรือเป็นเซนเซอร์อัตราฝนแบบตัวเลขที่ stale ระบบจะไม่สร้างค่าแห้ง
ปลอม การอ่านซ้ำ observation เดิมจะถูก deduplicate ก่อนเข้า pipeline เดิมของ `/reading`.
ดูสถานะได้ที่ `GET /ha-source/status` หรือ `GET /health`.
ใน status ให้ดู `last_prediction_ready`, `last_prediction_reason`, `ingest_count`
และ `prediction_count` แยกจาก `status=ready`: ถ้า source พร้อมแต่ข้อมูลต่อเนื่องยังไม่ถึง
หน้าต่างฟีเจอร์ จะเห็น `last_prediction_ready: false` และเหตุผล `waiting_for_history`.
ฟิลด์ `server_now_at_utc` ใช้เทียบเวลาของ add-on กับนาฬิกา HA เมื่อเวลาบนการ์ดไม่ตรงกัน.

การรันแบบ Docker Compose ภายนอก Supervisor ให้ตั้ง `HA_API_BASE` เป็น URL ของ Core API
และ `HA_TOKEN` เป็น long-lived token ผ่าน environment/secret เท่านั้น (ไม่ใส่ token ใน
`config.yaml` และระบบจะไม่แสดง token ใน log หรือ status). หากไม่กำหนด entity ครบหรือ
ข้อมูล stale ระบบจะคงข้อมูลเดิมไว้และรอรอบถัดไป; `POST /reading` ยังใช้เป็น fallback ได้เสมอ.

ตั้งแต่ v1.0.16 เป็นต้นไป reading ที่มาจาก HA จะเก็บ covariates ที่มีอยู่ลง MariaDB
และ runtime จะสร้าง illuminance/clear-sky feature พร้อมบันทึก model manifest
สำหรับตรวจ candidate แบบ shadow ได้ โดยยังไม่เปิดใช้ candidate กับ alert อัตโนมัติ
และ migration จะเพิ่มคอลัมน์ให้อัตโนมัติ ส่วน NWP hourly log จะเก็บ precipitation,
weather code, wind และ cloud พร้อม issue/valid time เพื่อทำ as-of join ตอนสร้าง
feature โดยไม่ให้ forecast ที่ออกภายหลังรั่วเข้าไปในประวัติ.

To backfill existing CSV history into MariaDB, run inside the add-on container:

```
python migrate_csv_to_db.py --data-dir /data/dataset
```

## Endpoints

- `GET /health`
- `GET /predict`
- `GET /predict?model=rf`
- `GET /forecast` — unified nowcast, radar arrival diagnostic, and hourly/daily NWP forecast
- `GET /official-warnings` — active TMD CAP warnings matched to `WEATHER_LAT/WEATHER_LON`
- `GET /rain-stop` — history-based experimental rain-stop estimate
- `GET /ha-source/status` — Home Assistant source configuration and health (token redacted)
- `POST /reading`

## Example POST Body

```json
{
  "temp": 30.1,
  "humidity": 70.2,
  "pressure": 1001.3,
  "rain_flag": 0
}
```

With timestamp:

```json
{
  "timestamp": "2026-05-13T10:30:00+07:00",
  "temp": 30.1,
  "humidity": 70.2,
  "pressure": 1001.3,
  "rain_flag": 0
}
```

CSV data is stored inside the add-on at `/data/dataset`.
