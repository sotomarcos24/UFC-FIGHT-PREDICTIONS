mkdir -p data
COMMIT=3d4886d222e72be2c5336b49b98b721f5103d949
BASE=https://raw.githubusercontent.com/Greco1899/scrape_ufc_stats/$COMMIT
for f in ufc_event_details ufc_fight_results ufc_fight_stats ufc_fighter_tott; do
  curl -sSfL "$BASE/$f.csv" -o "data/$f.csv"
done
ls -lh data