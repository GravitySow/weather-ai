#!/command/with-contenv bashio

set -e

export WEATHER_MODEL_KIND="$(bashio::config 'model_kind')"
export WEATHER_DATA_DIR="/data/dataset"
export DB_HOST="$(bashio::config 'db_host')"
export DB_PORT="$(bashio::config 'db_port')"
export DB_USER="$(bashio::config 'db_user')"
export DB_PASSWORD="$(bashio::config 'db_password')"
export DB_NAME="$(bashio::config 'db_name')"
export TELEGRAM_BOT_TOKEN="$(bashio::config 'telegram_bot_token')"
export TELEGRAM_CHAT_ID="$(bashio::config 'telegram_chat_id')"
export WEATHER_LAT="$(bashio::config 'weather_lat' '13.8692387')"
export WEATHER_LON="$(bashio::config 'weather_lon' '100.5180519')"
export WEATHER_TZ="$(bashio::config 'weather_tz' 'Asia/Bangkok')"
export NWP_BIAS_CORRECTION_PATH="$(bashio::config 'nwp_bias_correction_path' '')"
export NWP_TEMP_INTERVAL_PATH="$(bashio::config 'nwp_temp_interval_path' '')"
export ALERT_COOLDOWN_SECONDS="$(bashio::config 'alert_cooldown_seconds' '1800')"
export RAIN_STOP_CONFIRM_STREAK="$(bashio::config 'rain_stop_confirm_polls' '3')"
export REVISION_ALERT_ENABLED="$(bashio::config 'revision_alert_enabled' 'false')"
export REVISION_ALERT_COOLDOWN_SECONDS="$(bashio::config 'revision_alert_cooldown_seconds' '3600')"
export REVISION_POLL_SECONDS="$(bashio::config 'revision_poll_seconds' '900')"
export HA_SOURCE_ENABLED="$(bashio::config 'ha_source_enabled' 'false')"
export HA_TEMP_ENTITY="$(bashio::config 'ha_temp_entity' '')"
export HA_HUMIDITY_ENTITY="$(bashio::config 'ha_humidity_entity' '')"
export HA_PRESSURE_ENTITY="$(bashio::config 'ha_pressure_entity' '')"
export HA_RAIN_ENTITY="$(bashio::config 'ha_rain_entity' '')"
export HA_LIGHT_ENTITY="$(bashio::config 'ha_light_entity' '')"
export HA_POLL_SECONDS="$(bashio::config 'ha_poll_seconds' '60')"
export HA_STALE_AFTER_SECONDS="$(bashio::config 'ha_stale_after_seconds' '180')"
export HA_API_BASE="$(bashio::config 'ha_api_base' '')"

mkdir -p "${WEATHER_DATA_DIR}"

bashio::log.info "Starting Weather AI API"
bashio::log.info "Model kind: ${WEATHER_MODEL_KIND}"
bashio::log.info "Data directory: ${WEATHER_DATA_DIR}"

cd /app
exec uvicorn weather_api:app --host 0.0.0.0 --port 8000
