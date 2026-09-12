import React, { useState, useEffect } from 'react'
import axios from 'axios'
import { useNavigate } from 'react-router-dom'
import { Clock, Trash2, Shield, ShieldCheck, ShieldAlert, Activity } from 'lucide-react'

const History = () => {
  const [reviews, setReviews] = useState([])
  const [loading, setLoading] = useState(true)
  const navigate = useNavigate()

  const fetchHistory = async () => {
    try {
      const response = await axios.get('http://localhost:8000/api/reviews')
      setReviews(response.data)
    } catch (err) {
      console.error(err)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchHistory()
  }, [])

  const handleDelete = async (id) => {
    try {
      await axios.delete(`http://localhost:8000/api/reviews/${id}`)
      fetchHistory()
    } catch (err) {
      console.error(err)
    }
  }

  const getScoreIcon = (score) => {
    if (score >= 71) return <ShieldCheck className="h-5 w-5 text-green-600" />
    if (score >= 41) return <Shield className="h-5 w-5 text-yellow-500" />
    return <ShieldAlert className="h-5 w-5 text-red-600" />
  }

  if (loading) return <div className="flex justify-center mt-20"><Activity className="animate-spin text-indigo-500 h-8 w-8" /></div>

  return (
    <div className="space-y-6">
      <h2 className="text-2xl font-bold text-slate-800 flex items-center mb-6">
        <Clock className="mr-2 h-6 w-6 text-indigo-600" /> Analysis History
      </h2>

      {reviews.length === 0 ? (
        <div className="text-center text-slate-500 mt-20 bg-white p-12 rounded-2xl border border-slate-200">
          No reviews analyzed yet. Go to the Analyzer to get started.
        </div>
      ) : (
        <div className="bg-white rounded-2xl shadow-sm border border-slate-200 overflow-hidden">
          <ul className="divide-y divide-slate-200">
            {reviews.map((review) => (
              <li key={review.review_id} className="p-6 hover:bg-slate-50 transition-colors">
                <div className="flex items-start justify-between">
                  <div className="flex-1 mr-6">
                    <div className="flex items-center gap-3 mb-2">
                      {getScoreIcon(review.trust_score)}
                      <span className="font-bold text-lg">{review.trust_score}/100</span>
                      <span className="text-xs uppercase tracking-widest font-bold px-2 py-1 bg-slate-100 rounded text-slate-600">
                        {review.trust_level}
                      </span>
                      <span className="text-sm text-slate-500 ml-auto">
                        {new Date(review.timestamp).toLocaleString()}
                      </span>
                    </div>
                    <p className="text-slate-600 text-sm italic line-clamp-2">
                      "{review.review_text}"
                    </p>
                  </div>
                  <div className="flex flex-col gap-2">
                    <button
                      onClick={() => handleDelete(review.review_id)}
                      className="text-slate-400 hover:text-red-600 transition p-2"
                      title="Delete"
                    >
                      <Trash2 className="h-5 w-5" />
                    </button>
                  </div>
                </div>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  )
}

export default History
