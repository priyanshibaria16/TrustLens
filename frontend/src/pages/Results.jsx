import React, { useEffect } from 'react'
import { useLocation, useNavigate, useParams } from 'react-router-dom'
import { ShieldCheck, ShieldAlert, Shield, AlertTriangle, ThumbsUp, ThumbsDown, ArrowLeft, Info } from 'lucide-react'

const FeatureBar = ({ label, value, isNegative = false }) => {
  const percentage = Math.min(100, Math.max(0, value * 100))
  const barColor = isNegative 
    ? (percentage > 50 ? 'bg-red-500' : 'bg-yellow-400')
    : (percentage > 50 ? 'bg-green-500' : 'bg-indigo-400')
    
  return (
    <div className="mb-4">
      <div className="flex justify-between mb-1">
        <span className="text-sm font-medium text-slate-700">{label}</span>
        <span className="text-sm font-medium text-slate-500">{percentage.toFixed(0)}%</span>
      </div>
      <div className="w-full bg-slate-200 rounded-full h-2">
        <div className={`${barColor} h-2 rounded-full`} style={{ width: `${percentage}%` }}></div>
      </div>
    </div>
  )
}

const Results = () => {
  const location = useLocation()
  const navigate = useNavigate()
  const result = location.state?.result
  const originalText = location.state?.originalText

  useEffect(() => {
    if (!result) {
      navigate('/')
    }
  }, [result, navigate])

  if (!result) return null

  const getScoreColor = (score) => {
    if (score >= 71) return 'text-green-600'
    if (score >= 41) return 'text-yellow-500'
    return 'text-red-600'
  }

  const getScoreBg = (score) => {
    if (score >= 71) return 'bg-green-50 border-green-200'
    if (score >= 41) return 'bg-yellow-50 border-yellow-200'
    return 'bg-red-50 border-red-200'
  }

  const getScoreIcon = (score) => {
    if (score >= 71) return <ShieldCheck className="h-12 w-12 text-green-600" />
    if (score >= 41) return <Shield className="h-12 w-12 text-yellow-500" />
    return <ShieldAlert className="h-12 w-12 text-red-600" />
  }

  return (
    <div className="max-w-5xl mx-auto space-y-6">
      <button 
        onClick={() => navigate('/')} 
        className="flex items-center text-indigo-600 hover:text-indigo-800 text-sm font-medium transition"
      >
        <ArrowLeft className="h-4 w-4 mr-1" /> Back to Analyzer
      </button>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {/* Main Score Card */}
        <div className={`col-span-1 md:col-span-1 rounded-2xl shadow-sm border p-6 flex flex-col items-center justify-center text-center ${getScoreBg(result.trust_score)}`}>
          {getScoreIcon(result.trust_score)}
          <h2 className="text-xl font-bold text-slate-800 mt-4 uppercase tracking-wider">Trust Score</h2>
          <div className={`text-6xl font-black mt-2 ${getScoreColor(result.trust_score)}`}>
            {result.trust_score}
            <span className="text-2xl text-slate-400 font-medium">/100</span>
          </div>
          <div className={`mt-2 text-lg font-bold uppercase tracking-widest ${getScoreColor(result.trust_score)}`}>
            {result.trust_level}
          </div>
          
          <div className="mt-6 w-full border-t border-black/10 pt-4 flex justify-between px-4">
            <div className="text-center">
              <div className="text-xs text-slate-500 uppercase font-bold">Sentiment</div>
              <div className="font-semibold capitalize text-slate-800">{result.sentiment.label}</div>
            </div>
            <div className="text-center">
              <div className="text-xs text-slate-500 uppercase font-bold">Confidence</div>
              <div className="font-semibold text-slate-800">{(result.confidence * 100).toFixed(0)}%</div>
            </div>
          </div>
        </div>

        {/* Explanation & Contributors */}
        <div className="col-span-1 md:col-span-2 space-y-6">
          <div className="bg-white rounded-2xl shadow-sm border border-slate-200 p-6">
            <h3 className="text-lg font-bold text-slate-800 mb-2 flex items-center">
              <Info className="h-5 w-5 mr-2 text-indigo-500" />
              Why this score?
            </h3>
            <p className="text-slate-600 leading-relaxed text-lg">
              {result.explanation}
            </p>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <div className="bg-green-50 rounded-xl border border-green-100 p-5">
              <h4 className="font-bold text-green-800 flex items-center mb-3">
                <ThumbsUp className="h-4 w-4 mr-2" /> Positive Contributors
              </h4>
              {result.positive_contributors.length > 0 ? (
                <ul className="space-y-2">
                  {result.positive_contributors.map((c, i) => (
                    <li key={i} className="text-sm text-green-700 flex items-start">
                      <span className="text-green-500 mr-2 font-bold">+</span> {c}
                    </li>
                  ))}
                </ul>
              ) : (
                <div className="text-sm text-green-600/70 italic">No strong positive signals detected.</div>
              )}
            </div>
            
            <div className="bg-red-50 rounded-xl border border-red-100 p-5">
              <h4 className="font-bold text-red-800 flex items-center mb-3">
                <ThumbsDown className="h-4 w-4 mr-2" /> Negative Contributors
              </h4>
              {result.negative_contributors.length > 0 ? (
                <ul className="space-y-2">
                  {result.negative_contributors.map((c, i) => (
                    <li key={i} className="text-sm text-red-700 flex items-start">
                      <span className="text-red-500 mr-2 font-bold">-</span> {c}
                    </li>
                  ))}
                </ul>
              ) : (
                <div className="text-sm text-red-600/70 italic">No strong negative signals detected.</div>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Feature Breakdown */}
      <div className="bg-white rounded-2xl shadow-sm border border-slate-200 p-6">
        <h3 className="text-lg font-bold text-slate-800 mb-6">Feature Breakdown</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-x-12 gap-y-2">
          <div>
            <h4 className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-4 border-b pb-2">Review Quality Signals</h4>
            <FeatureBar label="Specificity & Concrete Details" value={result.features.specificity} />
            <FeatureBar label="Personal Experience" value={result.features.personal_experience} />
            <FeatureBar label="Supporting Evidence" value={result.features.evidence} />
            <FeatureBar label="Linguistic Quality" value={result.features.linguistic_quality} />
          </div>
          <div>
            <h4 className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-4 border-b pb-2">Suspicious Stylistic Signals</h4>
            <FeatureBar label="Promotional Language" value={result.features.promotion} isNegative={true} />
            <FeatureBar label="Exaggeration / Superlatives" value={result.features.exaggeration} isNegative={true} />
          </div>
        </div>
      </div>

      {/* Original Text Reference */}
      {originalText && (
        <div className="bg-slate-50 rounded-2xl border border-slate-200 p-6">
          <h3 className="text-sm font-bold text-slate-700 mb-3">Analyzed Text</h3>
          <p className="text-slate-600 text-sm italic border-l-4 border-slate-300 pl-4 py-1">
            "{originalText}"
          </p>
        </div>
      )}

      {/* Warning */}
      <div className="bg-blue-50 text-blue-800 rounded-xl p-4 flex items-start gap-3 border border-blue-100">
        <AlertTriangle className="h-5 w-5 text-blue-500 flex-shrink-0 mt-0.5" />
        <div className="text-sm">
          <strong>Important Limitation:</strong> {result.warnings[0]} The score is an estimate based on observable linguistic and stylistic patterns derived from the Ott Deceptive Opinion Spam dataset.
        </div>
      </div>
    </div>
  )
}

export default Results
