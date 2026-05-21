# OpenRouter Cloudflare Worker

This Worker is the private OpenRouter proxy for the Njx'Log search assistant.
The public Hugo site calls this Worker, and the Worker calls OpenRouter with the
secret API key stored in Cloudflare.

Do not put `OPENROUTER_API_KEY` in frontend code, Hugo config, or committed
files.

## Files

```text
wrangler.jsonc        Cloudflare Worker config
src/index.js          Worker request handler
.dev.vars.example     Local secret template
```

## Configuration

Non-secret settings live in `wrangler.jsonc`:

- `OPENROUTER_BASE_URL`: `https://openrouter.ai/api/v1`
- `OPENROUTER_MODEL`: default model for the assistant
- `ALLOWED_ORIGINS`: comma-separated browser origins allowed to call `/ask`
- `SITE_URL` and `SITE_TITLE`: metadata sent to OpenRouter

The API key must be stored as a Cloudflare secret named
`OPENROUTER_API_KEY`.

## Local Development

Install dependencies:

```powershell
cd workers/openrouter-proxy
npm install
```

Create local secrets:

```powershell
Copy-Item .dev.vars.example .dev.vars
```

Then edit `.dev.vars` and set your OpenRouter key locally.

Run the Worker:

```powershell
npm run dev
```

Health check:

```powershell
Invoke-RestMethod http://127.0.0.1:8787/health
```

## Deploy

Log in to Cloudflare:

```powershell
npx wrangler login
```

Set the production secret:

```powershell
npm run secret:set
```

Paste the OpenRouter key when prompted. Wrangler stores it as an encrypted
Cloudflare secret; it is not written into the repo.

Deploy:

```powershell
npm run deploy
```

The deployed endpoint will be:

```text
https://njx-log-ai-assistant.<your-workers-subdomain>.workers.dev/ask
```

Use that URL as `params.aiAssistant.endpoint` in `hugo.yaml`, then set
`params.aiAssistant.enabled` to `true`.

## Request Shape

```json
{
  "question": "What is this blog about?",
  "sources": [
    {
      "title": "Post title",
      "url": "https://njx-njx.github.io/posts/example/",
      "excerpt": "Relevant excerpt from index.json"
    }
  ],
  "history": [
    {
      "role": "user",
      "content": "Previous question"
    },
    {
      "role": "assistant",
      "content": "Previous answer"
    }
  ]
}
```

Response:

```json
{
  "answer": "Assistant answer",
  "model": "openai/gpt-4o-mini",
  "usage": null
}
```
