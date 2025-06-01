import React from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import {
  AppBar,
  Toolbar,
  Typography,
  Button,
  Box,
  IconButton,
} from '@mui/material';
import {
  Dashboard as DashboardIcon,
  CloudUpload as UploadIcon,
  ModelTraining as TrainingIcon,
  Psychology as InferenceIcon,
  Storage as ModelsIcon,
  Business as LawFirmsIcon,
  Description as TemplatesIcon,
  Comment as CommentsIcon,
} from '@mui/icons-material';

const Navbar = () => {
  const navigate = useNavigate();
  const location = useLocation();

  const navItems = [
    { path: '/dashboard', label: 'Dashboard', icon: <DashboardIcon /> },
    { path: '/contract-review', label: 'Contract Review', icon: <UploadIcon /> },
    { path: '/review-history', label: 'Review History', icon: <TrainingIcon /> },
    { path: '/law-firms', label: 'Law Firms', icon: <LawFirmsIcon /> },
    { path: '/templates', label: 'Templates', icon: <TemplatesIcon /> },
    { path: '/contract-comments', label: 'Comments', icon: <CommentsIcon /> },
  ];

  return (
    <AppBar position="static" elevation={0} sx={{ borderBottom: '1px solid #e0e0e0' }}>
      <Toolbar>
        <Typography variant="h6" component="div" sx={{ flexGrow: 1, fontWeight: 700 }}>
          📋 Contract Review Platform
        </Typography>
        
        <Box sx={{ display: 'flex', gap: 1 }}>
          {navItems.map((item) => (
            <Button
              key={item.path}
              color="inherit"
              startIcon={item.icon}
              onClick={() => navigate(item.path)}
              sx={{
                backgroundColor: location.pathname === item.path ? 'rgba(255,255,255,0.1)' : 'transparent',
                '&:hover': {
                  backgroundColor: 'rgba(255,255,255,0.1)',
                },
              }}
            >
              {item.label}
            </Button>
          ))}
        </Box>
      </Toolbar>
    </AppBar>
  );
};

export default Navbar; 