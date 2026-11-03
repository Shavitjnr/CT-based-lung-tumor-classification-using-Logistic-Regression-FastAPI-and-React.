import React from 'react';
import { Activity, UploadCloud, BarChart3, Info, CheckCircle2, ShieldAlert } from 'lucide-react';

interface NavbarProps {
  activeTab: string;
  setActiveTab: (tab: string) => void;
  hasResult: boolean;
}

export const Navbar: React.FC<NavbarProps> = ({ activeTab, setActiveTab, hasResult }) => {
  const navItems = [
    { id: 'home', label: 'Home Dashboard', icon: Activity },
    { id: 'upload', label: 'Analyze CT Scan', icon: UploadCloud },
    ...(hasResult ? [{ id: 'results', label: 'Prediction Results', icon: ShieldAlert }] : []),
    { id: 'performance', label: 'Model Performance', icon: BarChart3 },
    { id: 'about', label: 'About Project', icon: Info },
  ];

  return (
    <header className="bg-slate-900 border-b border-slate-800 sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Logo & Brand */}
          <div 
            className="flex items-center space-x-3 cursor-pointer select-none"
            onClick={() => setActiveTab('home')}
          >
            <div className="w-10 h-10 bg-teal-600 flex items-center justify-center text-white font-bold text-lg border border-teal-500">
              LS
            </div>
            <div>
              <div className="flex items-center space-x-2">
                <span className="text-xl font-bold tracking-tight text-white font-mono">LungScan<span className="text-teal-400">AI</span></span>
                <span className="text-[10px] tracking-wider uppercase font-semibold bg-teal-950 text-teal-300 border border-teal-800 px-1.5 py-0.5">
                  v1.0
                </span>
              </div>
              <p className="text-[11px] text-slate-400 -mt-0.5 tracking-wide">AI-Assisted Lung Tumor Classification</p>
            </div>
          </div>

          {/* Navigation Links */}
          <nav className="flex items-center space-x-1 sm:space-x-2">
            {navItems.map((item) => {
              const Icon = item.icon;
              const isActive = activeTab === item.id;
              return (
                <button
                  key={item.id}
                  onClick={() => setActiveTab(item.id)}
                  className={`flex items-center space-x-2 px-3.5 py-2 text-xs font-semibold tracking-wide border transition-all ${
                    isActive
                      ? 'bg-teal-600 text-white border-teal-500'
                      : 'text-slate-300 border-transparent hover:text-white hover:bg-slate-800 hover:border-slate-700'
                  }`}
                >
                  <Icon className="w-4 h-4" />
                  <span className="hidden md:inline">{item.label}</span>
                </button>
              );
            })}
          </nav>

          {/* System Status Pill */}
          <div className="hidden lg:flex items-center space-x-2 px-3 py-1.5 bg-slate-950 border border-slate-800 text-slate-300 text-xs font-mono">
            <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
            <span className="text-slate-400">Model:</span>
            <span className="text-teal-300 font-medium">LogisticRegression (IQ-OTH)</span>
          </div>
        </div>
      </div>
    </header>
  );
};

