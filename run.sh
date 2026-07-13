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
export WEATHER_LAT="$(bashio::config 'weather_lat' '13.8591')"
export WEATHER_LON="$(bashio::config 'weather_lon' '100.5217')"
export WEATHER_TZ="$(bashio::config 'weather_tz' 'Asia/Bangkok')"

mkdir -p "${WEATHER_DATA_DIR}"

bashio::log.info "Starting Weather AI API"
bashio::log.info "Model kind: ${WEATHER_MODEL_KIND}"
bashio::log.info "Data directory: ${WEATHER_DATA_DIR}"

cd /app
exec uvicorn weather_api:app --host 0.0.0.0 --port 8000
