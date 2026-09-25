import React, { useState, useEffect } from 'react'
import axios from 'axios'
import { BookOpen, Database, Settings, BarChart2 } from 'lucide-react'

const Methodology = () => {
  const [data, setData] = useState(null)
  
  useEffect(() => {
    const fetchMethodology = async () => {
      try {
        const response = await axios.get('http://localhost:8000/api/methodology')
        setData(response.data)
      } catch (err) {
        console.error(err)
      }
    }
    fetchMethodology()
  }, [])

  if (!data) return null

  return (
    <div className="max-w-4xl mx-auto space-y-8">
      <div className="text-center mb-10">
        <h2 className="text-3xl font-bold text-slate-800 flex items-center justify-center mb-4">
          <BookOpen className="mr-3 h-8 w-8 text-indigo-600" /> Research Methodology
        </h2>
        <p className="text-slate-500 text-lg">TrustLens is an explainable NLP framework developed exclusively using the Ott Deceptive Opinion Spam Corpus.</p>
      </div>

      <div className="space-y-6">
        <div className="bg-white rounded-2xl shadow-sm border border-slate-200 p-6 flex gap-4 items-start">
          <div className="bg-blue-100 p-3 rounded-xl flex-shrink-0">
            <Database className="h-6 w-6 text-blue-600" />
          </div>
          <div>
            <h3 className="text-lg font-bold text-slate-800 mb-2">1. Dataset</h3>
            <p className="text-slate-600 leading-relaxed">
              {data.dataset}. The dataset contains labeled positive and negative reviews that were generated truthfully by real customers or deceptively via crowdsourcing platforms. This balanced dataset is used as the primary supervised learning signal.
            </p>
          </div>
        </div>

        <div className="bg-white rounded-2xl shadow-sm border border-slate-200 p-6 flex gap-4 items-start">
          <div className="bg-indigo-100 p-3 rounded-xl flex-shrink-0">
            <Settings className="h-6 w-6 text-indigo-600" />
          </div>
          <div>
            <h3 className="text-lg font-bold text-slate-800 mb-2">2. Preprocessing & Feature Engineering</h3>
            <p className="text-slate-600 leading-relaxed mb-3">
              {data.preprocessing}.
            </p>
            <p className="text-slate-600 leading-relaxed">
              <strong>Extracted Features:</strong> {data.feature_extraction}
            </p>
          </div>
        </div>

        <div className="bg-white rounded-2xl shadow-sm border border-slate-200 p-6 flex gap-4 items-start">
          <div className="bg-green-100 p-3 rounded-xl flex-shrink-0">
            <BarChart2 className="h-6 w-6 text-green-600" />
          </div>
          <div>
            <h3 className="text-lg font-bold text-slate-800 mb-2">3. Modeling & Trust Score</h3>
            <p className="text-slate-600 leading-relaxed mb-3">
              <strong>Machine Learning:</strong> {data.modeling}. Models are trained on an 80/20 train-test split using Stratified K-Fold cross validation to avoid data leakage.
            </p>
            <p className="text-slate-600 leading-relaxed border-t pt-3 mt-3">
              <strong>Trust Score Engine:</strong> {data.trust_score} It is conceptually modeled as: <br />
              <code className="bg-slate-100 px-2 py-1 rounded text-sm mt-2 inline-block">Trust = Base + Positive Signals - Negative Signals - ML Signal</code>
            </p>
          </div>
        </div>
      </div>
      
      <div className="mt-8 bg-slate-50 border border-slate-200 rounded-2xl p-6 text-center text-sm text-slate-500">
        <p>
          TEXTUAL TRUSTWORTHINESS ≠ FACTUAL TRUTH<br/>
          This system identifies textual indicators associated with deceptive writing styles. It does not possess knowledge of factual events and cannot definitively label a review as completely true or false.
        </p>
      </div>
    </div>
  )
}

export default Methodology
