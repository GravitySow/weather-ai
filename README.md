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
and (optionally) a MariaDB database, returning 5m/10m/30m rain predictions from the
trained model, and pushing rain-alert notifications to Telegram when the rain state
changes.

## Home Assistant entities

On every `/reading` POST, the add-on pushes state updates straight into Home Assistant
via the Supervisor's Core API proxy (`homeassistant_api: true` in `config.yaml` — no
MQTT broker needed):

- `sensor.weather_ai_temperature`, `sensor.weather_ai_humidity`, `sensor.weather_ai_pressure`
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

A ready-made dashboard is in `dashboard.yaml` (HA's `sections` view type, requires HA
2024.9+) — a hero card (current temp/humidity/heat index + rain status in Thai), gauges for
the 5m/10m/30m rain probability, glance cards for current conditions and pressure trend/dew
point, an alert-status card, and 24h history graphs. Uses only stock HA cards, no HACS
required.

To install: **Settings → Dashboards → + Add Dashboard → New dashboard from scratch**, then
open the new dashboard's three-dot menu → **Edit Dashboard → three-dot menu → Raw configuration
editor**, and paste in the contents of `dashboard.yaml`.

## Configuration options

- `db_host`, `db_port`, `db_user`, `db_password`, `db_name` — MariaDB connection.
  Leave blank to skip DB persistence (CSV storage still works either way).
- `telegram_bot_token`, `telegram_chat_id` — Telegram Bot API credentials. Leave
  blank to disable Telegram notifications.

Rain alerts are sent to Telegram only on state changes (rain starts/stops, a new
rain-alert horizon triggers, or an alert clears) — not on every reading — to avoid
spamming the chat every minute.

To backfill existing CSV history into MariaDB, run inside the add-on container:

```
python migrate_csv_to_db.py --data-dir /data/dataset
```

## Endpoints

- `GET /health`
- `GET /predict`
- `GET /predict?model=rf`
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
