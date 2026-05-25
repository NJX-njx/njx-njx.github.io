# AGENTS.md

## Cursor Cloud specific instructions

This is a Hugo static blog ("Njx'Log") using the PaperMod theme (git submodule) and an optional Cloudflare Worker AI proxy.

### Services

| Service | Command | Port | Notes |
|---|---|---|---|
| Hugo dev server | `hugo server -D` | 1313 | Primary service; `--bind 0.0.0.0` for remote access |
| Cloudflare Worker | `cd workers/openrouter-proxy && npm run dev` | 8787 | Optional; requires `.dev.vars` with `OPENROUTER_API_KEY` |

### Quick checks / lint / test

All commands are documented in `README.md` under **Quick Checks**:

```
node --check assets/js/search-assistant.js
node --check scripts/search_quality.js
python3 -m py_compile scripts/llm_model.py
hugo --gc --minify
node scripts/search_quality.js public/index.json
```

The search quality harness requires a built `public/index.json` (run `hugo --gc --minify` first).

### Gotchas

- The Hugo config is `hugo.yaml` (not `.toml`). Do not create `hugo.toml`.
- Do **not** edit files under `themes/PaperMod/`; use Hugo's override system in root `layouts/` and `assets/`.
- The search page uses a custom layout named `search-assistant` (not PaperMod's built-in `search` layout) to avoid script conflicts.
- Hugo Extended is required (not the standard build) because the theme uses SCSS.
