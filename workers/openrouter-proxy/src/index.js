const SYSTEM_PROMPT = `You are the AI assistant for Njx'Log, a technical blog by Jiaxing Ni.

Answer based on the provided blog excerpts first. If the excerpts do not contain enough information, say so clearly and then give a concise general answer.

Style:
- Match the user's language.
- Be clear, concise, and technical.
- Cite relevant source titles when you use blog excerpts.
- Do not claim that you read blog content that was not provided.`;

const DEFAULT_ALLOWED_ORIGINS = [
  'https://njx-njx.github.io',
  'http://localhost:1313',
  'http://127.0.0.1:1313'
];

export default {
  async fetch(request, env) {
    const origin = request.headers.get('Origin') || '';
    const cors = corsHeaders(origin, env);

    if (request.method === 'OPTIONS') {
      return new Response(null, { status: 204, headers: cors });
    }

    const url = new URL(request.url);

    if (request.method === 'GET' && url.pathname === '/health') {
      return json({ ok: true }, { headers: cors });
    }

    if (request.method !== 'POST' || url.pathname !== '/ask') {
      return json({ error: 'Not found' }, { status: 404, headers: cors });
    }

    if (!isAllowedOrigin(origin, env)) {
      return json({ error: 'Origin not allowed' }, { status: 403, headers: cors });
    }

    if (!env.OPENROUTER_API_KEY) {
      return json({ error: 'Worker is missing OPENROUTER_API_KEY' }, { status: 500, headers: cors });
    }

    let payload;
    try {
      payload = await request.json();
    } catch {
      return json({ error: 'Invalid JSON body' }, { status: 400, headers: cors });
    }

    const validation = validatePayload(payload);
    if (!validation.ok) {
      return json({ error: validation.error }, { status: 400, headers: cors });
    }

    const messages = buildMessages(payload);
    const upstreamBody = {
      model: env.OPENROUTER_MODEL || 'openai/gpt-4o-mini',
      messages,
      temperature: 0.3,
      max_completion_tokens: 900
    };

    const baseUrl = trimTrailingSlash(env.OPENROUTER_BASE_URL || 'https://openrouter.ai/api/v1');
    const upstream = await fetch(`${baseUrl}/chat/completions`, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${env.OPENROUTER_API_KEY}`,
        'Content-Type': 'application/json',
        'HTTP-Referer': env.SITE_URL || 'https://njx-njx.github.io/',
        'X-Title': env.SITE_TITLE || "Njx'Log"
      },
      body: JSON.stringify(upstreamBody)
    });

    const responseText = await upstream.text();
    let responseJson = null;
    try {
      responseJson = responseText ? JSON.parse(responseText) : null;
    } catch {
      responseJson = null;
    }

    if (!upstream.ok) {
      return json({
        error: 'OpenRouter request failed',
        status: upstream.status,
        detail: responseJson?.error?.message || responseText.slice(0, 500)
      }, { status: upstream.status, headers: cors });
    }

    const answer = responseJson?.choices?.[0]?.message?.content || '';
    return json({
      answer,
      model: responseJson?.model || upstreamBody.model,
      usage: responseJson?.usage || null
    }, { headers: cors });
  }
};

function buildMessages(payload) {
  const sources = normalizeSources(payload.sources);
  const history = normalizeHistory(payload.history);
  const sourceContext = sources.length > 0
    ? sources.map((source, index) => {
      return [
        `Source ${index + 1}: ${source.title}`,
        `URL: ${source.url}`,
        `Excerpt: ${source.excerpt}`
      ].join('\n');
    }).join('\n\n')
    : 'No matching blog excerpts were provided.';

  return [
    { role: 'system', content: SYSTEM_PROMPT },
    ...history,
    {
      role: 'user',
      content: [
        `Blog excerpts:\n${sourceContext}`,
        `User question:\n${payload.question.trim()}`
      ].join('\n\n---\n\n')
    }
  ];
}

function validatePayload(payload) {
  if (!payload || typeof payload !== 'object') {
    return { ok: false, error: 'Body must be a JSON object' };
  }

  if (typeof payload.question !== 'string' || payload.question.trim().length === 0) {
    return { ok: false, error: 'question is required' };
  }

  if (payload.question.length > 1200) {
    return { ok: false, error: 'question is too long' };
  }

  if (payload.sources && !Array.isArray(payload.sources)) {
    return { ok: false, error: 'sources must be an array' };
  }

  if (payload.history && !Array.isArray(payload.history)) {
    return { ok: false, error: 'history must be an array' };
  }

  return { ok: true };
}

function normalizeSources(sources) {
  return (Array.isArray(sources) ? sources : []).slice(0, 3).map((source) => ({
    title: limitString(source?.title || 'Untitled', 160),
    url: limitString(source?.url || source?.permalink || '', 300),
    excerpt: limitString(source?.excerpt || source?.content || source?.summary || '', 1200)
  }));
}

function normalizeHistory(history) {
  return (Array.isArray(history) ? history : [])
    .filter((message) => message && ['user', 'assistant'].includes(message.role))
    .slice(-6)
    .map((message) => ({
      role: message.role,
      content: limitString(String(message.content || ''), 1200)
    }))
    .filter((message) => message.content.trim().length > 0);
}

function isAllowedOrigin(origin, env) {
  if (!origin) return false;
  return allowedOrigins(env).includes(origin);
}

function allowedOrigins(env) {
  const configured = typeof env.ALLOWED_ORIGINS === 'string'
    ? env.ALLOWED_ORIGINS.split(',').map((item) => item.trim()).filter(Boolean)
    : [];
  return configured.length > 0 ? configured : DEFAULT_ALLOWED_ORIGINS;
}

function corsHeaders(origin, env) {
  const headers = {
    'Access-Control-Allow-Methods': 'GET,POST,OPTIONS',
    'Access-Control-Allow-Headers': 'Content-Type',
    'Access-Control-Max-Age': '86400',
    'Vary': 'Origin'
  };

  if (isAllowedOrigin(origin, env)) {
    headers['Access-Control-Allow-Origin'] = origin;
  }

  return headers;
}

function json(data, init = {}) {
  return new Response(JSON.stringify(data), {
    ...init,
    headers: {
      'Content-Type': 'application/json; charset=utf-8',
      ...(init.headers || {})
    }
  });
}

function limitString(value, maxLength) {
  const text = String(value || '').trim();
  return text.length > maxLength ? `${text.slice(0, maxLength)}...` : text;
}

function trimTrailingSlash(value) {
  return String(value || '').replace(/\/+$/, '');
}
