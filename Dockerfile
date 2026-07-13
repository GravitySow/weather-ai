ARG BUILD_FROM=ghcr.io/hassio-addons/debian-base/amd64:stable
FROM ${BUILD_FROM}

ARG BUILD_ARCH
ARG BUILD_VERSION
ARG MODEL_BASE_URL

LABEL \
  io.hass.version="${BUILD_VERSION}" \
  io.hass.type="app" \
  io.hass.arch="${BUILD_ARCH}"

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV WEATHER_DATA_DIR=/data/dataset
ENV VIRTUAL_ENV=/opt/weather-ai-venv
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"

WORKDIR /app

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN apt-get update \
  && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    libgomp1 \
    python3 \
    python3-pip \
    python3-venv \
  && python3 -m venv "${VIRTUAL_ENV}" \
  && pip3 install --no-cache-dir --upgrade pip setuptools wheel \
  && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt

COPY run.sh /run.sh
COPY predict_weather_ai.py weather_api.py weather_features_lib.py weather_db.py telegram_notifier.py telegram_bot.py nwp_forecast.py migrate_csv_to_db.py ha_publisher.py radar_nowcast.py /app/
COPY weather_features.joblib weather_thresholds.joblib /app/
COPY weather_xgb_model_5m.joblib weather_xgb_model_10m.joblib weather_xgb_model_30m.joblib /app/

# Large RF/temp models aren't committed to the repo (see .gitignore) — pulled from
# a GitHub Release at build time instead, so the git repo (and every HA Supervisor
# clone/rebuild) stays small.
RUN test -n "${MODEL_BASE_URL}" \
  && for f in weather_rf_model_5m.joblib weather_rf_model_10m.joblib weather_rf_model_30m.joblib weather_temp_model_30m.joblib; do \
    curl -fsSL --retry 3 --retry-delay 2 -o "/app/${f}" "${MODEL_BASE_URL}/${f}"; \
  done

RUN chmod a+x /run.sh

EXPOSE 8000

CMD ["/run.sh"]
