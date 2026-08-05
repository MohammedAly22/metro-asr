# Publishing the report

`docs/index.html` is a static page — inline CSS, audio embedded as base64 data URIs, no build
step, no server, no dependency on any other file in the repo. It's named `index.html` rather
than `report.html` specifically so it serves as the site root when GitHub Pages publishes the
`/docs` folder. Publishing it means committing it to GitHub and turning on GitHub Pages once.

> [!IMPORTANT]
> Always build the committed copy with `--embed-audio` (see below). GitHub Pages only publishes
> the configured folder (`/docs`) — it does **not** serve the rest of the repository, so a
> relatively-linked `../test_samples/*.wav` resolves to a URL outside the published site and
> 404s once the page is actually hosted, even though it works fine when you open the file
> straight off disk from a full checkout. Embedding audio sidesteps this entirely.

## One-time setup

1. Commit and push `docs/`, and `test_samples/` to `main`:

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
   https://<your-username>.github.io/<your-repo>/
   ```

   **Important — this URL is lowercase**, regardless of how your GitHub username or repo name
   are capitalized. `MohammedAly22/Metro-ASR` publishes to
   `https://mohammedaly22.github.io/metro-asr/`, *not*
   `https://mohammedaly22.github.io/Metro-ASR/` — the capitalized form 404s even though the site
   is live. This is the actual URL the "🔊 Demo" badge in the README points to. Confirm your own
   URL from **Settings → Pages** itself (it shows a "Visit site" link once built), or query it
   directly:

   ```bash
   curl -s https://api.github.com/repos/<owner>/<repo>/pages | grep html_url
   ```

No custom domain, no Jekyll config, no Actions workflow needed — `/docs` on Pages serves the
files as-is, and `index.html` doesn't use any Jekyll templating.

## Re-publishing after you retrain a head or add clips

The report is generated from measured data, not hand-written, so re-publishing means
regenerating both JSON and HTML and pushing them:

```bash
python scripts/compare_lm_heads.py --out docs/results.json   # re-run greedy + every head, re-score
python scripts/build_report.py --embed-audio                  # docs/results.json -> docs/index.html

git add docs/results.json docs/index.html
git commit -m "Refresh report with updated decoding results"
git push
```

GitHub Pages redeploys automatically on every push to the branch/folder configured above —
nothing else to trigger.

## Local development

`--embed-audio` re-encodes and inlines every clip, which takes a few seconds and produces a
~14 MB file — fine for the one copy you commit, slow to regenerate on every edit while you're
iterating on the page layout itself. Drop the flag while you work on `scripts/build_report.py`:

```bash
python scripts/build_report.py           # fast rebuild; audio links to ../test_samples/
```

This smaller build only plays audio correctly when opened from inside a full checkout (the
relative link needs a sibling `test_samples/` directory) — that's fine on disk, but don't commit
this version as `docs/index.html`, since it's exactly the version that 404s once published. Add
`--embed-audio` back for the copy you actually commit and push.

## Troubleshooting

**Page 404s even though Settings → Pages shows "Deploy from a branch" saved and a green build.**
Almost certainly the URL casing issue described above — GitHub Pages serves project sites at the
lowercased `owner/repo` path even when the repository name itself has capitals. Check the exact
`html_url` via `GET /repos/<owner>/<repo>/pages` (needs a token with `repo` scope if the response
is 404 for an anonymous request — see below) rather than guessing the case from the repo name.

**`GET /repos/<owner>/<repo>/pages` returns 404 even for a public repo.** This endpoint 404s for
anonymous requests when the Pages *site visibility* is set to private (a paid-plan feature,
independent of the repository's own public/private setting) — GitHub returns 404 rather than 403
so it doesn't leak whether a private site exists. Authenticate the request
(`-H "Authorization: Bearer <token>"` with a token that has `repo` scope) to see the real status,
and check **Settings → Pages → Visibility** if a private/public toggle is present there.
