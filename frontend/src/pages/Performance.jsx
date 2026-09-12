import React, { useState, useEffect } from 'react'
import axios from 'axios'
import { Activity, Server, FileText, CheckCircle } from 'lucide-react'

const Performance = () => {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    const fetchData = async () => {
      try {
        const response = await axios.get('http://localhost:8000/api/model-performance')
        if (response.data.error) {
          setError(response.data.message)
        } else {
          setData(response.data)
        }
      } catch (err) {
        setError('Failed to fetch performance data.')
      } finally {
        setLoading(false)
      }
    }
    fetchData()
  }, [])

  if (loading) return <div className="flex justify-center mt-20"><Activity className="animate-spin text-indigo-500 h-8 w-8" /></div>
  
  if (error) return <div className="text-center text-red-500 mt-20 bg-red-50 p-6 rounded-xl">{error}</div>
  if (!data) return null

  return (
    <div className="space-y-8">
      <div className="text-center">
        <h2 className="text-3xl font-bold text-slate-800 flex items-center justify-center mb-2">
          <Server className="mr-3 h-8 w-8 text-indigo-600" /> Model Performance
        </h2>
        <p className="text-slate-500">Evaluation results on the test set (20% holdout)</p>
      </div>

      <div className="bg-white rounded-2xl shadow-sm border border-slate-200 p-6">
        <h3 className="text-lg font-bold text-slate-800 mb-6 flex items-center">
          <CheckCircle className="h-5 w-5 text-green-500 mr-2" /> Model Comparison
        </h3>
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-slate-200">
            <thead>
              <tr>
                <th className="px-6 py-3 bg-slate-50 text-left text-xs font-medium text-slate-500 uppercase tracking-wider">Model</th>
                <th className="px-6 py-3 bg-slate-50 text-left text-xs font-medium text-slate-500 uppercase tracking-wider">Accuracy</th>
                <th className="px-6 py-3 bg-slate-50 text-left text-xs font-medium text-slate-500 uppercase tracking-wider">Precision</th>
                <th className="px-6 py-3 bg-slate-50 text-left text-xs font-medium text-slate-500 uppercase tracking-wider">Recall</th>
                <th className="px-6 py-3 bg-slate-50 text-left text-xs font-medium text-slate-500 uppercase tracking-wider">F1 Score</th>
                <th className="px-6 py-3 bg-slate-50 text-left text-xs font-medium text-slate-500 uppercase tracking-wider">ROC-AUC</th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-slate-200">
              {data.comparison.map((model, idx) => (
                <tr key={idx} className={model.Model === data.metadata.model_name ? "bg-indigo-50/50" : ""}>
                  <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-slate-900">
                    {model.Model} {model.Model === data.metadata.model_name && "(Selected)"}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-slate-500">{(model.Accuracy * 100).toFixed(2)}%</td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-slate-500">{(model.Precision * 100).toFixed(2)}%</td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-slate-500">{(model.Recall * 100).toFixed(2)}%</td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-slate-500 font-bold">{(model.F1 * 100).toFixed(2)}%</td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-slate-500">{(model['ROC-AUC'] * 100).toFixed(2)}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="bg-white rounded-2xl shadow-sm border border-slate-200 p-6">
          <h3 className="text-lg font-bold text-slate-800 mb-6 flex items-center">
            <FileText className="h-5 w-5 text-indigo-500 mr-2" /> Current Model Details
          </h3>
          <ul className="space-y-4 text-sm text-slate-600">
            <li className="flex flex-col"><span className="font-semibold text-slate-800">Best Model:</span> {data.metadata.model_name}</li>
            <li className="flex flex-col"><span className="font-semibold text-slate-800">Dataset:</span> Ott Deceptive Opinion Spam Corpus</li>
            <li className="flex flex-col"><span className="font-semibold text-slate-800">Text Features:</span> {data.metadata.features.text}</li>
            <li className="flex flex-col"><span className="font-semibold text-slate-800">Linguistic Features:</span> {data.metadata.features.linguistic.length} custom features</li>
            <li className="flex flex-col"><span className="font-semibold text-slate-800">Trained On:</span> {new Date(data.metadata.training_date).toLocaleString()}</li>
          </ul>
        </div>
        
        <div className="bg-white rounded-2xl shadow-sm border border-slate-200 p-6">
           <h3 className="text-lg font-bold text-slate-800 mb-6 flex items-center">
            <Activity className="h-5 w-5 text-indigo-500 mr-2" /> Evaluation Note
          </h3>
          <p className="text-sm text-slate-600 leading-relaxed">
            The models were trained purely on textual characteristics. Precision and Recall reflect the model's ability to distinguish deceptive patterns as labeled in the Ott dataset. 
            <br/><br/>
            <strong>Important:</strong> High model confidence indicates strong alignment with known stylistic indicators of deception, not a factual verification of truth.
          </p>
        </div>
      </div>
    </div>
  )
}

export default Performance
