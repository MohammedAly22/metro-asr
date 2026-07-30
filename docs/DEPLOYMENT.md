# Publishing the report

`docs/report.html` is a static, self-contained page (audio links to `../test_samples/`,
everything else is inline CSS — no build step, no server). Publishing it means committing it
to GitHub and turning on GitHub Pages once.

## One-time setup

1. Commit and push `docs/report.html`, `docs/results.json` and `test_samples/` to `main`:

   ```bash
   git add docs/ test_samples/
   git commit -m "Add language-head comparison report"
   git push
   ```

2. On GitHub: **Settings → Pages**.
3. Under **Build and deployment → Source**, choose **Deploy from a branch**.
4. Under **Branch**, choose `main` and folder **`/docs`**, then **Save**.
5. GitHub builds and publishes within a minute or two. The report ends up at:

   ```
   https://<your-username>.github.io/<your-repo>/report.html
   ```

   For this repository that's `https://mohammedaly22.github.io/Metro-ASR/report.html` — the
   exact URL the "🔊 Demo" badge in the README points to. If your GitHub username or repo name
   differs, update that badge link in [README.md](../README.md) to match.

No custom domain, no Jekyll config, no Actions workflow needed — `/docs` on Pages serves the
files as-is, and `report.html` doesn't use any Jekyll templating.

## Re-publishing after you retrain a head or add clips

The report is generated from measured data, not hand-written, so re-publishing means
regenerating both JSON and HTML and pushing them:

```bash
python scripts/compare_lm_heads.py --out docs/results.json   # re-run greedy + every head, re-score
python scripts/build_report.py                                # docs/results.json -> docs/report.html

git add docs/results.json docs/report.html
git commit -m "Refresh report with updated decoding results"
git push
```

GitHub Pages redeploys automatically on every push to the branch/folder configured above —
nothing else to trigger.

## Viewing it without Pages

The file works standalone:

```bash
python scripts/build_report.py --embed-audio   # inlines audio as base64 — bigger file, portable
```

Open `docs/report.html` directly in a browser, or open it from a checkout of this repository
(the default, non-`--embed-audio` build links to `../test_samples/*.wav`, which only resolves
when the file is viewed from inside the repo — either on disk or on Pages).
