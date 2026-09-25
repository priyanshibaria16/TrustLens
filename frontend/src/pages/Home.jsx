import React, { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import axios from 'axios'
import { Shield, AlertCircle } from 'lucide-react'

const Home = () => {
  const [text, setText] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const navigate = useNavigate()

  const handleAnalyze = async () => {
    if (!text.trim()) {
      setError('Please enter a review to analyze.')
      return
    }
    setLoading(true)
    setError('')
    try {
      const response = await axios.post('http://localhost:8000/api/analyze', {
        review_text: text
      })
      if (response.data.success) {
        navigate(`/results/${response.data.review_id}`, { state: { result: response.data, originalText: text } })
      }
    } catch (err) {
      setError('An error occurred while analyzing the review. Make sure the backend is running.')
      console.error(err)
    } finally {
      setLoading(false)
    }
  }

  const loadExample = (type) => {
    if (type === 'high') {
      setText("I stayed at this hotel for four nights with my family. The room was clean and the staff responded quickly when the air conditioner stopped working on the second evening. Breakfast had reasonable options, although the coffee was disappointing. The hotel is about ten minutes from the main market.")
    } else if (type === 'low') {
      setText("AMAZING!!! BEST HOTEL EVER!!! 100% PERFECT!!! EVERYONE MUST STAY HERE!!! THE BEST HOTEL IN THE WORLD!!! ABSOLUTELY PERFECT!!!")
    } else {
      setText("The hotel was good overall. The room was clean and the location was convenient. The staff were friendly, although the breakfast could have had more options.")
    }
    setError('')
  }

  return (
    <div className="max-w-4xl mx-auto">
      <div className="text-center mb-10">
        <h1 className="text-4xl font-extrabold text-slate-900 tracking-tight sm:text-5xl mb-4">
          Understand why a review deserves your <span className="text-indigo-600">trust.</span>
        </h1>
        <p className="text-xl text-slate-500">
          TrustLens uses Explainable AI to analyze textual characteristics and identify deceptive signals in online reviews.
        </p>
      </div>

      <div className="bg-white rounded-2xl shadow-xl border border-slate-100 overflow-hidden">
        <div className="p-6 sm:p-8">
          <label htmlFor="review" className="block text-sm font-medium text-slate-700 mb-2">
            Paste an online review here
          </label>
          <textarea
            id="review"
            rows={8}
            className="block w-full rounded-xl border-slate-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 sm:text-lg p-4 bg-slate-50 transition-colors"
            placeholder="I recently stayed at this hotel and..."
            value={text}
            onChange={(e) => setText(e.target.value)}
          />
          
          {error && (
            <div className="mt-4 p-3 bg-red-50 text-red-700 rounded-lg flex items-center gap-2">
              <AlertCircle className="h-5 w-5" />
              <span>{error}</span>
            </div>
          )}

          <div className="mt-6 flex flex-col sm:flex-row gap-4 justify-between items-center">
            <div className="flex gap-2">
              <span className="text-sm text-slate-500 self-center mr-2">Examples:</span>
              <button onClick={() => loadExample('high')} className="px-3 py-1 text-xs font-medium bg-green-50 text-green-700 rounded-full hover:bg-green-100 transition">High Signal</button>
              <button onClick={() => loadExample('moderate')} className="px-3 py-1 text-xs font-medium bg-yellow-50 text-yellow-700 rounded-full hover:bg-yellow-100 transition">Moderate</button>
              <button onClick={() => loadExample('low')} className="px-3 py-1 text-xs font-medium bg-red-50 text-red-700 rounded-full hover:bg-red-100 transition">Low Signal</button>
            </div>
            
            <button
              onClick={handleAnalyze}
              disabled={loading}
              className="w-full sm:w-auto inline-flex items-center justify-center px-8 py-3 border border-transparent text-base font-medium rounded-xl text-white bg-indigo-600 hover:bg-indigo-700 md:py-4 md:text-lg md:px-10 transition-all shadow-md hover:shadow-lg disabled:opacity-70"
            >
              {loading ? (
                <>
                  <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                  </svg>
                  Analyzing...
                </>
              ) : (
                <>
                  <Shield className="mr-2 h-5 w-5" />
                  Analyze Review
                </>
              )}
            </button>
          </div>
        </div>
        <div className="bg-slate-50 px-6 py-4 border-t border-slate-100 flex items-center justify-center gap-2 text-sm text-slate-500">
          <AlertCircle className="h-4 w-4" />
          <span>Note: TrustLens estimates textual trustworthiness, not factual truth.</span>
        </div>
      </div>
    </div>
  )
}

export default Home
