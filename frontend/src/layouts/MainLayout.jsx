import React from 'react'
import { Link, useLocation } from 'react-router-dom'
import { ShieldCheck, Activity, Clock, BarChart2, BookOpen } from 'lucide-react'

const MainLayout = ({ children }) => {
  const location = useLocation()
  
  const navItems = [
    { name: 'Analyzer', path: '/', icon: ShieldCheck },
    { name: 'History', path: '/history', icon: Clock },
    { name: 'Analytics', path: '/analytics', icon: BarChart2 },
    { name: 'Performance', path: '/performance', icon: Activity },
    { name: 'Methodology', path: '/methodology', icon: BookOpen }
  ]

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900 font-sans flex flex-col">
      <header className="bg-white border-b border-slate-200 sticky top-0 z-10 shadow-sm">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between h-16">
            <div className="flex items-center">
              <Link to="/" className="flex items-center gap-2">
                <ShieldCheck className="h-8 w-8 text-indigo-600" />
                <div>
                  <span className="text-xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-indigo-600 to-blue-500">TRUSTLENS</span>
                  <span className="block text-[10px] text-slate-500 font-medium uppercase tracking-wider -mt-1">NLP Framework</span>
                </div>
              </Link>
            </div>
            <nav className="flex space-x-8">
              {navItems.map((item) => {
                const Icon = item.icon
                const isActive = location.pathname === item.path || (item.path !== '/' && location.pathname.startsWith(item.path))
                return (
                  <Link
                    key={item.name}
                    to={item.path}
                    className={`inline-flex items-center px-1 pt-1 border-b-2 text-sm font-medium transition-colors ${
                      isActive 
                        ? 'border-indigo-500 text-indigo-600' 
                        : 'border-transparent text-slate-500 hover:text-slate-800 hover:border-slate-300'
                    }`}
                  >
                    <Icon className="h-4 w-4 mr-2" />
                    {item.name}
                  </Link>
                )
              })}
            </nav>
          </div>
        </div>
      </header>

      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {children}
      </main>
      
      <footer className="bg-white border-t border-slate-200 mt-auto">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6 flex justify-between items-center text-sm text-slate-500">
          <div>TrustLens Academic Project &copy; {new Date().getFullYear()}</div>
          <div className="text-xs">Based exclusively on the Ott Deceptive Opinion Spam Corpus</div>
        </div>
      </footer>
    </div>
  )
}

export default MainLayout
