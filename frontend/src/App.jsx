import React from 'react'
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'
import MainLayout from './layouts/MainLayout'
import Home from './pages/Home'
import Results from './pages/Results'
import History from './pages/History'
import Analytics from './pages/Analytics'
import Performance from './pages/Performance'
import Methodology from './pages/Methodology'

function App() {
  return (
    <Router>
      <MainLayout>
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/results/:id" element={<Results />} />
          <Route path="/history" element={<History />} />
          <Route path="/analytics" element={<Analytics />} />
          <Route path="/performance" element={<Performance />} />
          <Route path="/methodology" element={<Methodology />} />
        </Routes>
      </MainLayout>
    </Router>
  )
}

export default App
