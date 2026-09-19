#!/bin/bash
# Monitor FOA completion and update results when done
# Run: nohup bash monitor_foa.sh &

PAPER_DIR="/data/yuehan/outputs/ucdpa_tta/paper"
SUMMARY_DIR="/data/yuehan/outputs/ucdpa_tta/results/summaries"
PYTHON="/data/yuehan/envs/tta_env/bin/python"

while true; do
    FOA_COUNT=$(ls $SUMMARY_DIR/foa__*.csv 2>/dev/null | wc -l)
    echo "$(date): FOA completed $FOA_COUNT/19 corruptions"

    if [ "$FOA_COUNT" -ge 19 ]; then
        echo "FOA COMPLETE! Updating results..."

        # Calculate new mean
        NEW_MEAN=$($PYTHON -c "
import csv, glob
accs = []
for f in sorted(glob.glob('$SUMMARY_DIR/foa__*.csv')):
    with open(f) as fh:
        r = csv.DictReader(fh)
        for row in r:
            accs.append(float(row.get('acc', row.get('accuracy', 0))))
print(f'{sum(accs)/len(accs):.2f}')
")

        echo "FOA final mean: $NEW_MEAN%"

        # Update main.tex
        sed -i "s/\\\\newcommand{\\\\foaAcc}{.*}/\\\\newcommand{\\\\foaAcc}{$NEW_MEAN}/" $PAPER_DIR/main.tex

        # Update table to remove partial note
        sed -i 's/FOA result is based on.*corruptions (partial; full results pending)./FOA is a derivative-free forward-only adaptation baseline./' $PAPER_DIR/tables/table7_cross_backbone.tex
        sed -i 's/\\\\foaAcc\$\\^\\dagger\$/\\\\foaAcc/g' $PAPER_DIR/tables/table7_cross_backbone.tex
        sed -i '/Partial: 7\/19/d' $PAPER_DIR/tables/table7_cross_backbone.tex

        # Update experiments.tex
        sed -i 's/(\\foaAcc\\%, $+9.63$~pp; partial, 7\/19 corruptions)/(\\foaAcc\\%, $+9.63$~pp)/' $PAPER_DIR/experiments.tex

        # Recompile
        cd $PAPER_DIR
        pdflatex -interaction=nonstopmode main.tex > /dev/null 2>&1
        bibtex main > /dev/null 2>&1
        pdflatex -interaction=nonstopmode main.tex > /dev/null 2>&1
        pdflatex -interaction=nonstopmode main.tex > /dev/null 2>&1

        PAGES=$(pdfinfo main.pdf | grep Pages | awk '{print $2}')
        echo "Compilation done. Pages: $PAGES"
        echo "FOA update complete!"
        break
    fi

    sleep 300  # Check every 5 minutes
done
