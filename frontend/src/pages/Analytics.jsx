import React, { useState, useEffect } from 'react'
import axios from 'axios'
import { PieChart, Pie, Cell, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts'
import { Activity, Shield } from 'lucide-react'

const COLORS = ['#22c55e', '#eab308', '#ef4444', '#64748b']

const Analytics = () => {
  const [summary, setSummary] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const fetchAnalytics = async () => {
      try {
        const response = await axios.get('http://localhost:8000/api/analytics/summary')
        setSummary(response.data)
      } catch (err) {
        console.error("Failed to fetch analytics", err)
      } finally {
        setLoading(false)
      }
    }
    fetchAnalytics()
  }, [])

  if (loading) return <div className="flex justify-center mt-20"><Activity className="animate-spin text-indigo-500 h-8 w-8" /></div>
  
  if (!summary) return <div className="text-center text-slate-500 mt-20">Analytics data not available.</div>

  const trustData = [
    { name: 'High Trust', value: summary.high_trust_reviews },
    { name: 'Moderate Trust', value: summary.moderate_trust_reviews },
    { name: 'Low Trust', value: summary.low_trust_reviews },
  ]
  
  const sentimentData = [
    { name: 'Positive', value: summary.sentiment_distribution.positive },
    { name: 'Negative', value: summary.sentiment_distribution.negative },
    { name: 'Neutral', value: summary.sentiment_distribution.neutral },
    { name: 'Mixed', value: summary.sentiment_distribution.mixed },
  ]

  return (
    <div className="space-y-6">
      <h2 className="text-2xl font-bold text-slate-800 flex items-center">
        <Shield className="mr-2 h-6 w-6 text-indigo-600" /> Dashboard Analytics
      </h2>
      
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="bg-white rounded-2xl shadow-sm border border-slate-200 p-6 flex flex-col items-center justify-center">
          <div className="text-sm font-bold uppercase tracking-wider text-slate-500">Total Analyzed Reviews</div>
          <div className="text-6xl font-black text-slate-800 mt-2">{summary.total_reviews}</div>
        </div>
        
        <div className="bg-white rounded-2xl shadow-sm border border-slate-200 p-6 flex flex-col items-center justify-center">
          <div className="text-sm font-bold uppercase tracking-wider text-slate-500">Average Trust Score</div>
          <div className="text-6xl font-black text-indigo-600 mt-2">{summary.average_trust_score}<span className="text-2xl text-slate-400">/100</span></div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-white rounded-2xl shadow-sm border border-slate-200 p-6">
          <h3 className="text-lg font-bold text-slate-800 mb-6">Trust Level Distribution</h3>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={trustData}
                  cx="50%"
                  cy="50%"
                  innerRadius={60}
                  outerRadius={80}
                  paddingAngle={5}
                  dataKey="value"
                  label
                >
                  {trustData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip />
                <Legend />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="bg-white rounded-2xl shadow-sm border border-slate-200 p-6">
          <h3 className="text-lg font-bold text-slate-800 mb-6">Sentiment Distribution</h3>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={sentimentData}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} />
                <XAxis dataKey="name" axisLine={false} tickLine={false} />
                <YAxis axisLine={false} tickLine={false} />
                <Tooltip cursor={{fill: '#f8fafc'}} />
                <Bar dataKey="value" radius={[4, 4, 0, 0]}>
                  {sentimentData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    </div>
  )
}

export default Analytics
