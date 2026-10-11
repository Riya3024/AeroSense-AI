
import { useCallback, useEffect, useMemo, useState } from 'react'
import {
  Activity,
  AlertTriangle,
  ArrowUpRight,
  CheckCircle2,
  Cloud,
  CloudRain,
  Database,
  Download,
  FileSearch,
  Gauge,
  RefreshCw,
  Search,
  ShieldCheck,
  Signal,
  Wind,
  X,
} from 'lucide-react'
import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import './App.css'

const API = 'http://localhost:8000'

type Page = 'overview' | 'events' | 'sensors' | 'model'

type EventRecord = {
  event_id: string
  sensor_id: string | number
  event_type: string
  network_context: string
  regional_event_supported: boolean
  start_time: string
  peak_time: string
  end_time?: string
  peak_pm25: number
  peak_final_score: number
  severity: string
  confidence_tier: string
  event_description: string
}


type HistoricalForecast = {
  status: string
  forecast_horizon_hours: number
  predicted_pm25: number
  unit: string
  model: string
  input_type: string
  live_sensor_data_used: boolean
  source: string
  source_timestamp: string | null
}

type Summary = {
  status: string
  event_count: number
  regional_event_supported_count: number
  event_type_counts: Record<string, number>
  severity_counts: Record<string, number>
  date_range: { start: string; end: string }
  peak_pm25_max: number
  source: string
}

type QualityReport = {
  status: string
  source: string
  records_checked: number
  records_with_quality_issues: number
  quality_issue_count: number
  issue_counts_by_reason: Record<string, number>
  suspicious_record_count: number
  suspicious_reading_count: number
  outlier_threshold_ug_m3: number
  suspicious_counts_by_reason: Record<string, number>
  issues: Array<{
    row_number: number
    event_id: string | null
    reason: string
  }>
  suspicious_readings: Array<{
    row_number: number
    event_id: string | null
    field: string
    value: number
    reason: string
  }>
  source_modified: boolean
}

type ApiResult<T> = {
  data: T | null
  error: string
}

type EventsResponse = {
  items: EventRecord[]
  pagination: { total: number }
}

async function fetchJson<T>(url: string): Promise<ApiResult<T>> {
  try {
    const response = await fetch(url)

    if (!response.ok) {
      throw new Error(`API returned HTTP ${response.status}`)
    }

    return { data: (await response.json()) as T, error: '' }
  } catch (error) {
    return {
      data: null,
      error:
        error instanceof Error
          ? error.message
          : 'Unable to connect to the API',
    }
  }
}

function formatDate(value?: string) {
  if (!value) return '—'
  const date = new Date(value)
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString()
}

function App() {
  const [activePage, setActivePage] = useState<Page>('overview')
  const [summary, setSummary] = useState<Summary | null>(null)
  const [quality, setQuality] = useState<QualityReport | null>(null)
  const [qualityError, setQualityError] = useState('')
  const [events, setEvents] = useState<EventRecord[]>([])
  const [totalEvents, setTotalEvents] = useState(0)
  const [severity, setSeverity] = useState('ALL')
  const [search, setSearch] = useState('')
  const [modelStatus, setModelStatus] = useState<Record<string, unknown> | null>(null)
  const [cloudStatus, setCloudStatus] = useState<Record<string, unknown> | null>(null)
  const [health, setHealth] = useState<Record<string, unknown> | null>(null)
  const [apiError, setApiError] = useState('')
  const [loading, setLoading] = useState(true)
  const [selectedEvent, setSelectedEvent] = useState<EventRecord | null>(null)
  const [lastUpdated, setLastUpdated] = useState('')
  
  const [forecast, setForecast] = useState<HistoricalForecast | null>(null)
  const [forecastError, setForecastError] = useState('')
  const [forecastLoading, setForecastLoading] = useState(true)

  const loadDashboard = useCallback(async () => {
    const [summaryResult, qualityResult, modelResult, cloudResult, healthResult] =
      await Promise.all([
        fetchJson<Summary>(`${API}/api/intelligence/summary`),
        fetchJson<QualityReport>(`${API}/api/data-quality`),
        fetchJson<Record<string, unknown>>(`${API}/api/model/status`),
        fetchJson<Record<string, unknown>>(`${API}/api/cloud/status`),
        fetchJson<Record<string, unknown>>(`${API}/health`),
      ])

    setApiError('')

    if (summaryResult.data) {
      setSummary(summaryResult.data)
    } else {
      setApiError(`Summary: ${summaryResult.error}`)
    }

    if (qualityResult.data) {
      setQuality(qualityResult.data)
      setQualityError('')
    } else {
      setQualityError(qualityResult.error)
    }

    setModelStatus(modelResult.data)
    setCloudStatus(cloudResult.data)
    setHealth(healthResult.data)

    const query = new URLSearchParams()
    query.set('limit', '100')
    query.set('offset', '0')
    if (severity !== 'ALL') query.set('severity', severity)

    const eventsResult = await fetchJson<EventsResponse>(
      `${API}/api/events?${query.toString()}`,
    )

    if (eventsResult.data) {
      setEvents(eventsResult.data.items ?? [])
      setTotalEvents(eventsResult.data.pagination?.total ?? 0)
    } else {
      setApiError((previous) =>
        [previous, `Events: ${eventsResult.error}`].filter(Boolean).join(' | '),
      )
    }

    setLastUpdated(new Date().toLocaleTimeString())
    setLoading(false)
  }, [severity])

  useEffect(() => {
    let active = true
    const task = window.setTimeout(() => {
      if (active) void loadDashboard()
    }, 0)
    return () => {
      active = false
      window.clearTimeout(task)
    }
  }, [loadDashboard])

  
const loadForecast = useCallback(async () => {
  try {
    const result = await fetchJson<HistoricalForecast>(
      `${API}/api/prediction/latest`,
    )
    setForecastError('')

    if (result.data) {
      setForecast(result.data)
    } else {
      setForecastError(result.error || 'Historical forecast is unavailable')
    }
  } catch {
    setForecastError('Unable to load the historical forecast')
  } finally {
    setForecastLoading(false)
  }
}, [])

useEffect(() => {
  let active = true
  const task = window.setTimeout(() => {
    if (active) void loadForecast()
  }, 0)
  return () => {
    active = false
    window.clearTimeout(task)
  }
}, [loadForecast])


  const filteredEvents = useMemo(() => {
    const term = search.trim().toLowerCase()

    return events.filter((event) => {
      return (
        !term ||
        event.event_id.toLowerCase().includes(term) ||
        String(event.sensor_id).toLowerCase().includes(term) ||
        event.event_type.toLowerCase().includes(term) ||
        event.severity.toLowerCase().includes(term)
      )
    })
  }, [events, search])

  const severityCounts = summary?.severity_counts ?? {}

  
  const chartData = Object.entries(severityCounts).map(([name, count]) => ({
    name,
    events: count,
  }))

  const sensorStats = useMemo(() => {
    const sensors = new Map<string, {
      id: string
      events: number
      highestPm25: number
      highestSeverity: string
      regionalSupport: number
    }>()

    const severityRank: Record<string, number> = {
      LOW: 1,
      MODERATE: 2,
      HIGH: 3,
      SEVERE: 4,
      EXTREME: 5,
      CRITICAL: 6,
    }

    events.forEach((event) => {
      const id = String(event.sensor_id)
      const current = sensors.get(id)
      const pm25 = Number(event.peak_pm25) || 0

      if (!current) {
        sensors.set(id, {
          id,
          events: 1,
          highestPm25: pm25,
          highestSeverity: event.severity,
          regionalSupport: event.regional_event_supported ? 1 : 0,
        })
      } else {
        current.events += 1
        current.highestPm25 = Math.max(current.highestPm25, pm25)
        current.regionalSupport += event.regional_event_supported ? 1 : 0

        if (
          (severityRank[event.severity] ?? 0) >
          (severityRank[current.highestSeverity] ?? 0)
        ) {
          current.highestSeverity = event.severity
        }
      }
    })

    return Array.from(sensors.values()).sort(
      (a, b) => b.events - a.events,
    )
  }, [events])

  function downloadFile(filename: string, content: string, type: string) {
    const blob = new Blob([content], { type })
    const url = URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = url
    link.download = filename
    document.body.appendChild(link)
    link.click()
    link.remove()
    URL.revokeObjectURL(url)
  }

  function exportEvents() {
    const headers = [
      'event_id',
      'sensor_id',
      'event_type',
      'severity',
      'network_context',
      'peak_pm25',
      'peak_time',
    ]

    const rows = filteredEvents.map((event) =>
      headers
        .map((header) => {
          const value = event[header as keyof EventRecord] ?? ''
          return `"${String(value).replaceAll('"', '""')}"`
        })
        .join(','),
    )

    downloadFile(
      'aerosense-events.csv',
      [headers.join(','), ...rows].join('\n'),
      'text/csv;charset=utf-8;',
    )
  }

  function exportQuality() {
    if (!quality) return
    downloadFile(
      'aerosense-data-quality-report.json',
      JSON.stringify(quality, null, 2),
      'application/json;charset=utf-8;',
    )
  }

  const apiOnline = Boolean(summary)
  const capabilities =
    health?.capabilities &&
    typeof health.capabilities === 'object'
      ? (health.capabilities as Record<string, unknown>)
      : {}
  const predictionEnabled = capabilities.pm25_prediction === true

  
  const historicalPredictionAvailable = forecast?.status === 'success'

  const qualityPassed =
    quality?.status === 'passed' && quality.quality_issue_count === 0

  const navItems: { id: Page; label: string; icon: typeof Activity }[] = [
    { id: 'overview', label: 'Intelligence overview', icon: Activity },
    { id: 'events', label: 'Event monitoring', icon: AlertTriangle },
    { id: 'sensors', label: 'Sensor network', icon: Signal },
    { id: 'model', label: 'Model status', icon: Gauge },
  ]

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark"><Wind size={25} /></div>
          <div>
            <div className="brand-name">AeroSense<span>AI</span></div>
            <div className="brand-caption">ENVIRONMENTAL INTELLIGENCE</div>
          </div>
        </div>

        <div className="nav-label">WORKSPACE</div>

        <nav className="sidebar-nav" aria-label="Main navigation">
          {navItems.map(({ id, label, icon: Icon }) => (
            <button
              key={id}
              type="button"
              className={`nav-item ${activePage === id ? 'active' : ''}`}
              onClick={() => setActivePage(id)}
              aria-current={activePage === id ? 'page' : undefined}
            >
              <Icon size={18} />
              <span>{label}</span>
            </button>
          ))}
        </nav>

        <div className="sidebar-bottom">
          <div className="sidebar-status">
            <span className={`status-dot ${apiOnline ? 'online' : ''}`} />
            <div>
              <strong>Backend connection</strong>
              <small>{apiOnline ? 'API responding' : 'Connection pending'}</small>
            </div>
          </div>
          <div className="sidebar-foot">
            AEROSENSE RESEARCH BUILD
            <br />
            Version 0.1.0 · Local environment
          </div>
        </div>
      </aside>

      <main className="main-content">
        <header className="topbar">
          <div className="breadcrumb">
            Workspace <span>/</span>{' '}
            <strong>{navItems.find((item) => item.id === activePage)?.label}</strong>
          </div>
          <div className="topbar-right">
            <span className="environment-pill">
              <span className="status-dot online" />
              LOCAL ENVIRONMENT
            </span>
            <button
              type="button"
              className="icon-button"
              onClick={() => void loadDashboard()}
              aria-label="Refresh dashboard"
              disabled={loading}
            >
              <RefreshCw size={17} />
            </button>
          </div>
        </header>

        <div className="page-content">
          {apiError && (
            <div className="error-banner">
              <AlertTriangle size={18} />
              <div>
                <strong>API connection issue</strong>
                <p>{apiError}. Check that the Python API is running on port 8000.</p>
              </div>
              <button
                type="button"
                className="icon-button"
                onClick={() => void loadDashboard()}
                aria-label="Retry API connection"
              >
                <RefreshCw size={16} />
              </button>
            </div>
          )}

          <section className="page-heading">
            <div>
              <div className="eyebrow">
                <span className="eyebrow-line" />
                AIR QUALITY EVENT INTELLIGENCE
              </div>
              <h1>
                {activePage === 'overview' && <>Environmental <span>Intelligence</span></>}
                {activePage === 'events' && <>Event <span>Monitoring</span></>}
                {activePage === 'sensors' && <>Sensor <span>Network</span></>}
                {activePage === 'model' && <>Model <span>Status</span></>}
              </h1>
              <p>
                {activePage === 'overview' &&
                  'Monitor air-quality event data and review data-quality findings.'}
                {activePage === 'events' &&
                  'Search, filter, export, and inspect recorded events.'}
                {activePage === 'sensors' &&
                  'Explore sensor IDs represented in the available event records.'}
                {activePage === 'model' &&
                  'Review model capability and backend service responses.'}
              </p>
            </div>

            {activePage === 'events' && (
              <button
                type="button"
                className="button-secondary"
                onClick={exportEvents}
                disabled={!filteredEvents.length}
              >
                <Download size={16} /> Export visible events
              </button>
            )}
            {activePage === 'overview' && (
              <button
                type="button"
                className="button-secondary"
                onClick={exportQuality}
                disabled={!quality}
              >
                <Download size={16} /> Export quality report
              </button>
            )}
          </section>

          <section className="status-strip">
            <div className="status-strip-main">
              <span className={`status-dot ${apiOnline ? 'online' : ''}`} />
              <span>{apiOnline ? 'Research data connected' : 'Waiting for research data'}</span>
            </div>
            <span>Source: {summary?.source ?? 'Awaiting API'}</span>
            <span>Last refreshed: {lastUpdated || '—'}</span>
          </section>

          {activePage === 'overview' && (
            <>
              <section className="section-heading">
                <div>
                  <h2>Network overview</h2>
                  <p>
                    Summary of detected events in the dataset. These records do not
                    necessarily represent live sensor connections.
                  </p>
                </div>
                <span className="live-label">
                  <span className={`status-dot ${apiOnline ? 'online' : ''}`} />
                  {apiOnline ? 'API CONNECTED' : 'API DISCONNECTED'}
                </span>
              </section>

              <section className="metrics-grid">
                <article className="metric-card">
                  <div className="metric-top">
                    <span>Total detected events</span>
                    <div className="metric-icon blue"><Activity size={19} /></div>
                  </div>
                  <div className="metric-value">
                    {summary ? summary.event_count.toLocaleString() : '—'}
                  </div>
                  <div className="metric-foot">
                    <span className="metric-neutral"><Database size={13} /> Dataset records</span>
                    <span>Available history</span>
                  </div>
                </article>

                <article className="metric-card">
                  <div className="metric-top">
                    <span>High severity</span>
                    <div className="metric-icon amber"><AlertTriangle size={19} /></div>
                  </div>
                  <div className="metric-value">
                    {(severityCounts.HIGH ?? 0).toLocaleString()}
                  </div>
                  <div className="metric-foot">
                    <span className="metric-neutral"><ArrowUpRight size={13} /> Classified events</span>
                    <span>Severity label</span>
                  </div>
                </article>

                <article className="metric-card">
                  <div className="metric-top">
                    <span>Regional supported</span>
                    <div className="metric-icon green"><ShieldCheck size={19} /></div>
                  </div>
                  <div className="metric-value">
                    {summary ? summary.regional_event_supported_count.toLocaleString() : '—'}
                  </div>
                  <div className="metric-foot">
                    <span className="metric-neutral"><Signal size={13} /> Network corroboration</span>
                    <span>Evidence-based</span>
                  </div>
                </article>

                <article className="metric-card">
                  <div className="metric-top">
                    <span>Maximum recorded PM2.5</span>
                    <div className="metric-icon purple"><CloudRain size={19} /></div>
                  </div>
                  <div className="metric-value">
                    {summary ? summary.peak_pm25_max.toLocaleString() : '—'}
                    <small> µg/m³</small>
                  </div>
                  <div className="metric-foot">
                    <span className="metric-neutral"><Cloud size={13} /> Recorded maximum</span>
                    <span>Dataset value</span>
                  </div>
                </article>
              </section>



              
<section className="panel" style={{ marginBottom: 24 }}>
  <div className="section-heading">
    <div>
      <h2>One-hour PM2.5 forecast</h2>
      <p>Estimate based on the latest complete historical feature row.</p>
    </div>
    <button
      type="button"
      className="button-secondary"
      onClick={() => void loadForecast()}
      disabled={forecastLoading}
    >
      <RefreshCw size={16} />
      {forecastLoading ? 'Loading…' : 'Refresh forecast'}
    </button>
  </div>

  <div style={{
    padding: 20,
    border: '1px solid #253448',
    borderRadius: 12,
    background: '#101b2a',
    color: '#e8eef7',
  }}>
    <div style={{ color: '#aebed0', fontSize: 13 }}>
      Predicted PM2.5
    </div>

    <div style={{ fontSize: 36, fontWeight: 700, marginTop: 8 }}>
      {forecast
        ? forecast.predicted_pm25.toLocaleString(undefined, {
            maximumFractionDigits: 2,
          })
        : '—'}
      <span style={{ fontSize: 14, marginLeft: 8 }}>
        {forecast?.unit ?? 'µg/m³'}
      </span>
    </div>

    <p style={{ marginTop: 12 }}>
      <strong>Data status:</strong> Historical data — not live
    </p>
    <p>
      <strong>Forecast horizon:</strong>{' '}
      {forecast ? `${forecast.forecast_horizon_hours} hour` : '—'}
    </p>
    <p>
      <strong>Model:</strong> {forecast?.model ?? 'AeroSense XGBoost'}
    </p>
    <p>
      <strong>Source:</strong> {forecast?.source ?? '—'}
    </p>
    <p>
      <strong>Source timestamp:</strong>{' '}
      {forecast?.source_timestamp
        ? `${formatDate(forecast.source_timestamp)} (UTC)`
        : '—'}
    </p>
    <p>
      <strong>Live sensor data used:</strong>{' '}
      {forecast
        ? (forecast.live_sensor_data_used ? 'Yes' : 'No')
        : 'Not confirmed'}
    </p>
  </div>

  {forecastError && (
    <div className="error-banner" style={{ marginTop: 14 }}>
      <AlertTriangle size={18} />
      <div>
        <strong>Forecast unavailable</strong>
        <p>{forecastError}</p>
      </div>
      <button
        type="button"
        className="icon-button"
        onClick={() => void loadForecast()}
        aria-label="Retry forecast"
      >
        <RefreshCw size={16} />
      </button>
    </div>
  )}

  <p className="panel-note" style={{ marginTop: 12 }}>
    This is a model estimate from saved historical features, not a
    current ambient measurement or verified live forecast. Prediction
    errors may be substantial, especially during extreme pollution events.
  </p>
</section>


              <section className="section-heading">
                <div>
                  <h2>Data quality</h2>
                  <p>Validation checks and readings flagged for manual review.</p>
                </div>
              </section>

              {qualityError && !quality && (
                <div className="error-banner">
                  <AlertTriangle size={18} />
                  <div>
                    <strong>Data-quality report unavailable</strong>
                    <p>{qualityError}</p>
                  </div>
                </div>
              )}

              {quality && (
                <>
                  <div className="quality-status" style={{
                    display: 'flex',
                    alignItems: 'flex-start',
                    gap: 12,
                    padding: 18,
                    marginBottom: 18,
                    borderRadius: 12,
                    border: '1px solid var(--border-color, #dce5e8)',
                   background: '#101b2a',
                   color: '#e8eef7',
borderColor: '#253448',
                  }}>
                    {qualityPassed
                      ? <CheckCircle2 size={22} color="#16845b" />
                      : <AlertTriangle size={22} color="#b7791f" />}
                    <div>
                      <strong>{qualityPassed ? 'Validation checks passed' : 'Review data-quality findings'}</strong>
                      <p
  style={{
    margin: '5px 0 0',
    lineHeight: 1.6,
    color: '#aebed0',
  }}
>
                        Checked {quality.records_checked.toLocaleString()} records from {quality.source}.
                        {' '}{quality.quality_issue_count.toLocaleString()} validation issues were found.
                        {' '}Suspicious readings are listed separately and are not automatically removed.
                      </p>
                    </div>
                  </div>

                  <section className="metrics-grid">
                    <article className="metric-card">
                      <div className="metric-top"><span>Records checked</span><div className="metric-icon blue"><FileSearch size={19} /></div></div>
                      <div className="metric-value">{quality.records_checked.toLocaleString()}</div>
                      <div className="metric-foot"><span>Records inspected</span><span>{quality.source}</span></div>
                    </article>
                    <article className="metric-card">
                      <div className="metric-top"><span>Quality issues</span><div className="metric-icon green"><ShieldCheck size={19} /></div></div>
                      <div className="metric-value">{quality.quality_issue_count.toLocaleString()}</div>
                      <div className="metric-foot"><span>Validation findings</span><span>{quality.records_with_quality_issues} records affected</span></div>
                    </article>
                    <article className="metric-card">
                      <div className="metric-top"><span>Suspicious readings</span><div className="metric-icon amber"><AlertTriangle size={19} /></div></div>
                      <div className="metric-value">{quality.suspicious_reading_count.toLocaleString()}</div>
                      <div className="metric-foot"><span>Review recommended</span><span>{quality.suspicious_record_count} records flagged</span></div>
                    </article>
                    <article className="metric-card">
                      <div className="metric-top"><span>Review threshold</span><div className="metric-icon purple"><Gauge size={19} /></div></div>
                      <div className="metric-value">{quality.outlier_threshold_ug_m3.toLocaleString()}<small> µg/m³</small></div>
                      <div className="metric-foot"><span>PM2.5 threshold</span><span>Flagged, not deleted</span></div>
                    </article>
                  </section>

                  {quality.suspicious_readings.length > 0 && (
                    <section className="panel">
                      <div className="section-heading">
                        <div>
                          <h2>Readings requiring review</h2>
                          <p>Values above the configured review threshold.</p>
                        </div>
                      </div>
                      <div className="table-wrap">
                        <table className="events-table">
                          <thead><tr><th>Event ID</th><th>Row</th><th>Field</th><th>Reading (µg/m³)</th><th>Finding</th></tr></thead>
                          <tbody>
                            {quality.suspicious_readings.map((item, index) => (
                              <tr key={`${item.event_id}-${item.row_number}-${index}`}>
                                <td>{item.event_id ?? '—'}</td>
                                <td>{item.row_number}</td>
                                <td>{item.field}</td>
                                <td>{item.value.toLocaleString()}</td>
                                <td><span className="severity-pill severity-high">Review</span></td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                      <p className="panel-note">
                        A flagged value is not necessarily wrong. Confirm unusual readings against
                        sensor conditions and supporting evidence before taking action.
                      </p>
                    </section>
                  )}

                  {quality.quality_issue_count > 0 && (
                    <section className="panel">
                      <div className="section-heading"><div><h2>Validation findings</h2><p>Issues detected by the current checks.</p></div></div>
                      <div className="table-wrap">
                        <table className="events-table">
                          <thead><tr><th>Event ID</th><th>Row</th><th>Reason</th></tr></thead>
                          <tbody>
                            {quality.issues.map((item, index) => (
                              <tr key={`${item.row_number}-${index}`}>
                                <td>{item.event_id ?? '—'}</td><td>{item.row_number}</td><td>{item.reason}</td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    </section>
                  )}

                  <div className="status-strip">
                    <div className="status-strip-main">
                      <span className={`status-dot ${quality.source_modified ? '' : 'online'}`} />
                      <span>Source dataset {quality.source_modified ? 'modified' : 'not modified by quality checks'}</span>
                    </div>
                    <span>Status: {quality.status.replaceAll('_', ' ')}</span>
                  </div>
                </>
              )}

              <section className="section-heading">
                <div>
                  <h2>Event severity distribution</h2>
                  <p>Counts of event classifications in the dataset.</p>
                </div>
              </section>

              <section className="panel chart-panel">
                {chartData.length > 0 ? (
                  <div style={{ width: '100%', height: 320 }}>
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={chartData} margin={{ top: 12, right: 20, left: 0, bottom: 8 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#e5ebef" />
                        <XAxis dataKey="name" />
                        <YAxis allowDecimals={false} />
                        <Tooltip />
                        <Bar dataKey="events" name="Events" fill="#3386c8" radius={[5, 5, 0, 0]} />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                ) : (
                  <p>{loading ? 'Loading severity distribution…' : 'No chart data is available.'}</p>
                )}
              </section>
            </>
          )}

          {activePage === 'events' && (
            <>
              <section className="metrics-grid">
                <article className="metric-card">
                  <div className="metric-top"><span>Total matching events</span><div className="metric-icon blue"><Activity size={19} /></div></div>
                  <div className="metric-value">{totalEvents.toLocaleString()}</div>
                  <div className="metric-foot"><span>API results</span><span>Current severity filter</span></div>
                </article>
                <article className="metric-card">
                  <div className="metric-top"><span>Loaded in this view</span><div className="metric-icon purple"><Database size={19} /></div></div>
                  <div className="metric-value">{events.length.toLocaleString()}</div>
                  <div className="metric-foot"><span>Records loaded</span><span>Maximum 100</span></div>
                </article>
              </section>

              <section className="panel">
                <div className="event-controls" style={{ display: 'flex', flexWrap: 'wrap', gap: 12, alignItems: 'center', marginBottom: 18 }}>
                  <div className="search-field" style={{ display: 'flex', gap: 8, alignItems: 'center', flex: '1 1 220px' }}>
                    <Search size={17} />
                    <input
                      type="search"
                      value={search}
                      onChange={(event) => setSearch(event.target.value)}
                      placeholder="Search event, sensor, type, severity…"
                      aria-label="Search events"
                    />
                  </div>
                  <select value={severity} onChange={(event) => setSeverity(event.target.value)} aria-label="Filter events by severity">
                    <option value="ALL">All severities</option>
                    <option value="CRITICAL">Critical</option>
                    <option value="EXTREME">Extreme</option>
                    <option value="SEVERE">Severe</option>
                    <option value="HIGH">High</option>
                    <option value="MODERATE">Moderate</option>
                    <option value="LOW">Low</option>
                  </select>
                  <button type="button" className="button-secondary" onClick={exportEvents} disabled={!filteredEvents.length}>
                    <Download size={16} /> Export visible events
                  </button>
                </div>

                <div className="table-wrap">
                  <table className="events-table">
                    <thead><tr><th>Event</th><th>Sensor</th><th>Type</th><th>Severity</th><th>Peak PM2.5</th><th>Peak time</th><th>Details</th></tr></thead>
                    <tbody>
                      {filteredEvents.map((event) => (
                        <tr key={event.event_id}>
                          <td>{event.event_id}</td>
                          <td>{event.sensor_id}</td>
                          <td>{event.event_type}</td>
                          <td><span className={`severity-pill severity-${event.severity.toLowerCase()}`}>{event.severity}</span></td>
                          <td>{Number(event.peak_pm25).toLocaleString()}</td>
                          <td>{formatDate(event.peak_time)}</td>
                          <td><button type="button" className="button-secondary" onClick={() => setSelectedEvent(event)}>Inspect</button></td>
                        </tr>
                      ))}
                      {filteredEvents.length === 0 && (
                        <tr><td colSpan={7}>{loading ? 'Loading events…' : 'No events match your search and severity filter.'}</td></tr>
                      )}
                    </tbody>
                  </table>
                </div>
                <p className="panel-note">
                  Showing up to 100 records from the API response. Search and severity filters apply to these loaded records.
                </p>
              </section>
            </>
          )}

          {activePage === 'sensors' && (
            <>
              <div className="status-strip">
                <div className="status-strip-main">
                  <Signal size={16} />
                  <span>Dataset-derived sensor summary</span>
                </div>
                <span>This is not a live connectivity monitor.</span>
              </div>

              <section className="metrics-grid">
                <article className="metric-card">
                  <div className="metric-top"><span>Unique sensors loaded</span><div className="metric-icon blue"><Signal size={19} /></div></div>
                  <div className="metric-value">{sensorStats.length.toLocaleString()}</div>
                  <div className="metric-foot"><span>Unique sensor IDs</span><span>Loaded event records</span></div>
                </article>
                <article className="metric-card">
                  <div className="metric-top"><span>Events loaded</span><div className="metric-icon purple"><Database size={19} /></div></div>
                  <div className="metric-value">{events.length.toLocaleString()}</div>
                  <div className="metric-foot"><span>Event records</span><span>Maximum 100 loaded</span></div>
                </article>
                <article className="metric-card">
                  <div className="metric-top"><span>Regional support</span><div className="metric-icon green"><ShieldCheck size={19} /></div></div>
                  <div className="metric-value">{events.filter((event) => event.regional_event_supported).length.toLocaleString()}</div>
                  <div className="metric-foot"><span>Loaded records</span><span>Dataset field</span></div>
                </article>
              </section>

              <section className="panel">
                <div className="section-heading">
                  <div><h2>Sensor records</h2><p>Summary grouped by sensor ID from the loaded event records.</p></div>
                </div>
                <div className="table-wrap">
                  <table className="events-table">
                    <thead><tr><th>Sensor ID</th><th>Loaded events</th><th>Maximum PM2.5</th><th>Highest severity</th><th>Regionally supported events</th></tr></thead>
                    <tbody>
                      {sensorStats.map((sensor) => (
                        <tr key={sensor.id}>
                          <td>{sensor.id}</td>
                          <td>{sensor.events}</td>
                          <td>{sensor.highestPm25.toLocaleString()} µg/m³</td>
                          <td><span className={`severity-pill severity-${sensor.highestSeverity.toLowerCase()}`}>{sensor.highestSeverity}</span></td>
                          <td>{sensor.regionalSupport}</td>
                        </tr>
                      ))}
                      {sensorStats.length === 0 && <tr><td colSpan={5}>No sensor records are available.</td></tr>}
                    </tbody>
                  </table>
                </div>
                <p className="panel-note">
                  Sensor statistics are calculated from the events currently loaded by the API. They do not establish that sensors are online or transmitting now.
                </p>
              </section>
            </>
          )}

          {activePage === 'model' && (
            <>
              <section className="metrics-grid">
                <article className="metric-card">
                  <div className="metric-top"><span>PM2.5 prediction</span><div className="metric-icon blue"><Gauge size={19} /></div></div>
                  
<div className="metric-value">
  {forecastLoading
    ? 'Checking…'
    : historicalPredictionAvailable
      ? 'Available (historical)'
      : predictionEnabled
        ? 'Enabled'
        : 'Not enabled'}
</div>

                  <div className="metric-foot">
                    <span className="metric-neutral">Backend capability</span>
                    
<span>
  {historicalPredictionAvailable
    ? 'Historical prediction succeeded'
    : predictionEnabled
      ? 'Reported by health endpoint'
      : 'Health capability flag is disabled'}
</span>

                  </div>
                </article>
                <article className="metric-card">
                  <div className="metric-top"><span>Model endpoint</span><div className="metric-icon purple"><Activity size={19} /></div></div>
                  <div className="metric-value">{modelStatus ? 'Responding' : 'Unknown'}</div>
                  <div className="metric-foot"><span>Endpoint response</span><span>{modelStatus ? 'Status received' : 'No status response'}</span></div>
                </article>
                <article className="metric-card">
                  <div className="metric-top"><span>Cloud endpoint</span><div className="metric-icon blue"><Cloud size={19} /></div></div>
                  <div className="metric-value">{cloudStatus ? 'Responding' : 'Unknown'}</div>
                  <div className="metric-foot"><span>Endpoint response</span><span>{cloudStatus ? 'Status received' : 'No status response'}</span></div>
                </article>
              </section>

              <section className="panel">
                <div className="section-heading">
                  <div><h2>Capability details</h2><p>Information reported by the local backend health endpoint.</p></div>
                </div>
                <div style={{ padding: 20, lineHeight: 1.8 }}>
                  
<p>
  <strong>Health capability flag:</strong>{' '}
  {predictionEnabled
    ? 'The backend reports PM2.5 prediction as enabled.'
    : 'The backend reports PM2.5 prediction as disabled.'}
</p>
<p>
  <strong>Historical prediction endpoint:</strong>{' '}
  {forecastLoading
    ? 'Checking prediction result…'
    : historicalPredictionAvailable
      ? `Working. Latest prediction: ${forecast.predicted_pm25.toFixed(2)} ${forecast.unit}.`
      : forecastError
        ? `Unavailable: ${forecastError}`
        : 'No successful prediction result is currently available.'}
</p>
<p>
  <strong>Prediction data source:</strong>{' '}
  Historical engineered features; not live sensor data.
</p>

                  <p>
                    <strong>Model endpoint:</strong> {modelStatus ? 'Responding to requests.' : 'No response received.'}
                  </p>
                  <p>
                    <strong>Cloud endpoint:</strong> {cloudStatus ? 'Responding to requests.' : 'No response received.'}
                  </p>
                  <p>
                    A responding endpoint does not by itself prove that a trained model is loaded, accurate, or producing predictions.
                  </p>
                  <button type="button" className="button-secondary" onClick={() => {
  void loadDashboard()
  void loadForecast()
}} disabled={loading}>
                    <RefreshCw size={16} /> Refresh status
                  </button>
                </div>
              </section>
            </>
          )}

          <footer style={{ padding: '28px 0 12px', color: 'var(--muted, #6b7280)', fontSize: 12 }}>
            AeroSense AI · Local research dashboard · Data reflects the available dataset and API responses, not necessarily live environmental conditions.
          </footer>
        </div>
      </main>

      {selectedEvent && (
        <div className="modal-backdrop" role="presentation" onClick={() => setSelectedEvent(null)}>
          <section
            className="modal-card"
            role="dialog"
            aria-modal="true"
            aria-labelledby="event-detail-title"
            onClick={(event) => event.stopPropagation()}
          >
            <div className="modal-header">
              <div>
                <h2 id="event-detail-title">Event details</h2>
                <p>{selectedEvent.event_id}</p>
              </div>
              <button type="button" className="icon-button" onClick={() => setSelectedEvent(null)} aria-label="Close event details">
                <X size={18} />
              </button>
            </div>
            <dl className="detail-list">
              <dt>Sensor ID</dt><dd>{selectedEvent.sensor_id}</dd>
              <dt>Event type</dt><dd>{selectedEvent.event_type}</dd>
              <dt>Severity</dt><dd>{selectedEvent.severity}</dd>
              <dt>Network context</dt><dd>{selectedEvent.network_context || '—'}</dd>
              <dt>Regional support</dt><dd>{selectedEvent.regional_event_supported ? 'Supported' : 'Not supported'}</dd>
              <dt>Confidence tier</dt><dd>{selectedEvent.confidence_tier || '—'}</dd>
              <dt>Peak PM2.5</dt><dd>{Number(selectedEvent.peak_pm25).toLocaleString()} µg/m³</dd>
              <dt>Peak time</dt><dd>{formatDate(selectedEvent.peak_time)}</dd>
              <dt>Start time</dt><dd>{formatDate(selectedEvent.start_time)}</dd>
              <dt>Description</dt><dd>{selectedEvent.event_description || 'No description provided.'}</dd>
            </dl>
            <button type="button" className="button-secondary" onClick={() => setSelectedEvent(null)}>Close</button>
          </section>
        </div>
      )}
    </div>
  )
}

export default App
