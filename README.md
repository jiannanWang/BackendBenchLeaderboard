# BackendBench Leaderboard - React Application

A modern, responsive React web application for benchmarking and ranking GPU kernel implementations. This leaderboard platform enables researchers to submit, compare, and discover high-performance kernel implementations across different operators, DSLs, and hardware devices.

![BackendBench Leaderboard](https://via.placeholder.com/800x400?text=BackendBench+Leaderboard)

## 🚀 Features

### Interactive Leaderboard Navigation
- **Operator-based Navigation**: Click on specific operators to view dedicated ranking pages
- **Dynamic Routing**: Each operator has its own URL (e.g., `/operator/matmul`)
- **Performance Ranking**: Submissions ranked by TFLOPS performance with detailed metrics
- **Multi-view Support**: Toggle between table and card views for optimal data presentation
- **Advanced Filtering**: Filter by DSL framework and target device within each operator
- **Sorting Options**: Sort by performance, submission date, correctness score, or author

### Operator Categories
- **Matrix Multiplication** (`/operator/matmul`) - High-performance GEMM implementations
- **Attention** (`/operator/attention`) - Self-attention and multi-head attention kernels
- **Layer Normalization** (`/operator/layernorm`) - Fused layer normalization implementations
- **2D Convolution** (`/operator/conv2d`) - Optimized convolution operations
- **GELU Activation** (`/operator/gelu`) - Gaussian Error Linear Unit implementations
- **Softmax** (`/operator/softmax`) - Numerically stable softmax kernels
- **Linear Layer** (`/operator/linear`) - Fully connected layer implementations
- **Embedding** (`/operator/embedding`) - Token and positional embedding lookups

### Submission System
- **Easy Upload**: Drag-and-drop file upload with support for multiple file formats
- **Comprehensive Forms**: Detailed submission forms capturing all necessary metadata
- **Validation**: Built-in form validation and file type checking
- **Draft Support**: Save work in progress as drafts
- **Requirements Guide**: Clear submission guidelines and requirements

### User Experience
- **Modern React Architecture**: Component-based architecture with React Router
- **Responsive Design**: Optimized for desktop, tablet, and mobile devices
- **Interactive Elements**: Hover effects, modal dialogs, and smooth transitions
- **Accessibility**: Keyboard navigation and screen reader support
- **Performance**: Optimized loading and rendering for large datasets

## 🛠️ Technology Stack

- **Frontend Framework**: React 18.2.0
- **Routing**: React Router DOM 6.3.0
- **Styling**: CSS3 with CSS Grid, Flexbox, and CSS Variables
- **Icons**: Font Awesome 6.0
- **Fonts**: Inter (Google Fonts)
- **Build Tool**: Create React App
- **Package Manager**: npm

## 📋 Supported Operators

The leaderboard currently supports submissions for the following operators (focused on NanoGPT training):

- **Matrix Multiplication** - Dense linear algebra operations
- **Attention Mechanisms** - Self-attention and multi-head attention
- **Layer Normalization** - Stable training normalization
- **2D Convolution** - Computer vision workloads
- **GELU Activation** - Gaussian Error Linear Unit
- **Softmax** - Probability distribution normalization
- **Linear Layers** - Fully connected operations
- **Embedding** - Token and positional lookups

## 🔧 Supported DSLs & Frameworks

- **Triton**: GPU kernel programming language
- **CUDA**: NVIDIA's parallel computing platform
- **CUTLASS**: CUDA Templates for Linear Algebra Subroutines
- **PyTorch**: Machine learning framework
- **JAX**: Autograd and XLA
- **TVM**: Tensor compiler stack

## 🎯 Target Devices

- NVIDIA H100
- NVIDIA A100  
- NVIDIA V100
- NVIDIA RTX 4090
- NVIDIA RTX 3080
- Other GPU architectures (with manual specification)

## 📁 Project Structure

```
BackendBenchLeaderboard/
├── public/
│   └── index.html          # Main HTML template
├── src/
│   ├── components/         # Reusable React components
│   │   ├── Header.js       # Navigation header
│   │   ├── Footer.js       # Site footer
│   │   ├── OperatorCard.js # Operator selection cards
│   │   ├── LeaderboardTable.js # Performance table
│   │   └── SubmissionModal.js  # Detailed submission view
│   ├── pages/              # Main page components
│   │   ├── HomePage.js     # Main dashboard with operator grid
│   │   ├── OperatorPage.js # Individual operator leaderboards
│   │   └── SubmitPage.js   # Kernel submission form
│   ├── data/
│   │   └── mockData.js     # Sample data and utilities
│   ├── styles.css          # Main stylesheet
│   ├── App.js              # Main app component and routing
│   └── index.js            # React app entry point
├── package.json            # Dependencies and scripts
└── README.md              # This file
```

## 🚀 Getting Started

### Prerequisites
- Node.js (v14 or higher)
- npm or yarn

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/BackendBenchLeaderboard.git
   cd BackendBenchLeaderboard
   ```

2. **Install dependencies**
   ```bash
   npm install
   # or
   yarn install
   ```

3. **Start the development server**
   ```bash
   npm start
   # or
   yarn start
   ```

4. **Open your browser**
   Navigate to `http://localhost:3000`

### Build for Production

```bash
npm run build
# or
yarn build
```

## 💡 Usage Guide

### Browsing Operators
1. Open the application to view the main dashboard
2. Click on any operator card to view its specific leaderboard
3. Use the back button or navigation to return to the main page

### Viewing Operator Rankings
1. Select an operator from the main page
2. Filter submissions by DSL or device using the dropdown filters
3. Sort results by performance, date, correctness, or author
4. Toggle between table and card views
5. Click "View Details" on any submission for comprehensive metrics

### Submitting a Kernel
1. Navigate to the submission page via the header or operator page
2. Fill out the required kernel information
3. Upload your kernel files using drag-and-drop or file browser
4. Review submission requirements and submit for review

### Navigation Examples
- Main page: `http://localhost:3000/`
- Matrix multiplication leaderboard: `http://localhost:3000/operator/matmul`
- Attention leaderboard: `http://localhost:3000/operator/attention`
- Submission page: `http://localhost:3000/submit`

## 🎨 Design Features

### Component Architecture
- **Modular Components**: Reusable UI components with clear responsibilities
- **React Router Integration**: Seamless navigation between operator-specific pages
- **State Management**: Local component state with React hooks
- **Props-based Data Flow**: Clean data passing between components

### Visual Design
- **Dark Theme**: Professional dark background with blue and coral accents
- **Gradient Elements**: Subtle gradients for visual depth
- **Typography**: Inter font family for optimal readability
- **Consistent Spacing**: 8px grid system for uniform layout

### Interactive Elements
- **Clickable Operator Cards**: Direct navigation to operator-specific leaderboards
- **Hover Effects**: Subtle animations on buttons and cards
- **Loading States**: Visual feedback during operations
- **Modal Dialogs**: Detailed submission views in overlay modals
- **Smooth Scrolling**: Enhanced navigation experience

### Responsive Breakpoints
- **Desktop**: 1200px+ (full layout)
- **Tablet**: 768px-1199px (adapted layout)
- **Mobile**: <768px (stacked layout)

## 🔮 Future Enhancements

### Backend Integration
- RESTful API for submission management
- Database storage for persistent data
- User authentication and profiles
- Automated benchmarking pipeline

### Advanced Features
- Search functionality across all operators
- Performance trending over time
- Detailed comparison tools between submissions
- Community voting and reviews
- Integration with GitHub repositories
- Advanced analytics dashboard

### Technical Improvements
- Server-side rendering (Next.js migration)
- Progressive Web App (PWA) features
- Real-time updates with WebSocket integration
- Enhanced caching strategies

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes following the component structure
4. Test the application (`npm test`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

### Development Guidelines
- Follow React best practices and hooks patterns
- Use functional components with hooks
- Maintain consistent file naming and structure
- Add PropTypes or TypeScript for type safety
- Write unit tests for new components

## 📄 Scripts

- `npm start` - Start development server
- `npm run build` - Create production build
- `npm test` - Run test suite
- `npm run eject` - Eject from Create React App (irreversible)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Inspired by existing leaderboard platforms like GPUMode and SWE-bench
- Built for the BackendBench project to encourage kernel optimization research
- Design inspired by modern developer tools and platforms
- React community for excellent documentation and ecosystem

## 📞 Contact

For questions, suggestions, or contributions, please:
- Open an issue on GitHub
- Contact the BackendBench team
- Join our Discord community

---

**BackendBench Leaderboard** - Empowering GPU kernel optimization research through transparent benchmarking and community collaboration.