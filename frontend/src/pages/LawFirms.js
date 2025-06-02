import React, { useState, useEffect } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Button,
  Grid,
  Alert,
  TextField,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Chip,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  ListItemSecondaryAction,
  IconButton,
  Paper,
  Divider,
} from '@mui/material';
import {
  Business as BusinessIcon,
  Add as AddIcon,
  Delete as DeleteIcon,
  Edit as EditIcon,
  Label as LabelIcon,
  CalendarToday as CalendarIcon,
} from '@mui/icons-material';
import { useQuery, useMutation, useQueryClient } from 'react-query';
import axios from 'axios';

const API_BASE = process.env.REACT_APP_API_URL || 'http://localhost:9100';

function LawFirms() {
  const [showAddDialog, setShowAddDialog] = useState(false);
  const [formData, setFormData] = useState({
    name: '',
    keywords: ''
  });
  const [alert, setAlert] = useState(null);
  const queryClient = useQueryClient();

  // Fetch law firms
  const { data: lawFirmsData, isLoading, error } = useQuery(
    'law-firms',
    async () => {
      const response = await axios.get(`${API_BASE}/api/law-firms`);
      return response.data;
    },
    {
      refetchInterval: 30000,
    }
  );

  // Add law firm mutation
  const addMutation = useMutation(
    async (firmData) => {
      const keywords = firmData.keywords.split(',').map(k => k.trim()).filter(k => k);
      const response = await axios.post(`${API_BASE}/api/law-firms`, {
        name: firmData.name,
        keywords: keywords
      });
      return response.data;
    },
    {
      onSuccess: () => {
        queryClient.invalidateQueries('law-firms');
        setShowAddDialog(false);
        setFormData({ name: '', keywords: '' });
        setAlert({ type: 'success', message: 'Law firm added successfully!' });
        setTimeout(() => setAlert(null), 3000);
      },
      onError: (error) => {
        setAlert({ type: 'error', message: error.response?.data?.detail || 'Failed to add law firm' });
        setTimeout(() => setAlert(null), 5000);
      }
    }
  );

  const handleSubmit = () => {
    if (!formData.name.trim() || !formData.keywords.trim()) {
      setAlert({ type: 'error', message: 'Please fill in all required fields' });
      setTimeout(() => setAlert(null), 3000);
      return;
    }
    addMutation.mutate(formData);
  };

  const lawFirms = lawFirmsData?.law_firms || [];

  const formatDate = (dateString) => {
    return new Date(dateString).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    });
  };

  if (isLoading) {
    return (
      <Box sx={{ maxWidth: 1200, mx: 'auto', textAlign: 'center', py: 8 }}>
        <Typography variant="h6">Loading law firms...</Typography>
      </Box>
    );
  }

  return (
    <Box sx={{ maxWidth: 1200, mx: 'auto' }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Box>
          <Typography variant="h4" gutterBottom sx={{ fontWeight: 600 }}>
            Law Firms Management
          </Typography>
          <Typography variant="body1" color="text.secondary">
            Manage law firms and their detection keywords for contract analysis
          </Typography>
        </Box>
        <Button
          variant="contained"
          startIcon={<AddIcon />}
          onClick={() => setShowAddDialog(true)}
          size="large"
        >
          Add Law Firm
        </Button>
      </Box>

      {alert && (
        <Alert severity={alert.type} sx={{ mb: 3 }}>
          {alert.message}
        </Alert>
      )}

      {/* Statistics */}
      <Grid container spacing={3} sx={{ mb: 4 }}>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                <Box
                  sx={{
                    p: 1,
                    borderRadius: 1,
                    backgroundColor: 'primary.light',
                    color: 'primary.main',
                    mr: 2,
                  }}
                >
                  <BusinessIcon />
                </Box>
                <Typography variant="h4" sx={{ fontWeight: 600 }}>
                  {lawFirms.length}
                </Typography>
              </Box>
              <Typography variant="body2" color="text.secondary">
                Total Law Firms
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                <Box
                  sx={{
                    p: 1,
                    borderRadius: 1,
                    backgroundColor: 'info.light',
                    color: 'info.main',
                    mr: 2,
                  }}
                >
                  <LabelIcon />
                </Box>
                <Typography variant="h4" sx={{ fontWeight: 600 }}>
                  {lawFirms.reduce((total, firm) => total + firm.keywords.length, 0)}
                </Typography>
              </Box>
              <Typography variant="body2" color="text.secondary">
                Total Keywords
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Law Firms List */}
      {lawFirms.length > 0 ? (
        <Grid container spacing={3}>
          {lawFirms.map((firm) => (
            <Grid item xs={12} md={6} key={firm.id}>
              <Card sx={{ height: '100%' }}>
                <CardContent>
                  <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                    <BusinessIcon sx={{ color: 'primary.main', mr: 1 }} />
                    <Typography variant="h6" sx={{ flexGrow: 1 }}>
                      {firm.name}
                    </Typography>
                    <IconButton size="small" color="primary">
                      <EditIcon />
                    </IconButton>
                    <IconButton size="small" color="error">
                      <DeleteIcon />
                    </IconButton>
                  </Box>
                  
                  <Divider sx={{ mb: 2 }} />
                  
                  <Box sx={{ mb: 2 }}>
                    <Typography variant="subtitle2" gutterBottom sx={{ display: 'flex', alignItems: 'center' }}>
                      <LabelIcon sx={{ mr: 1, fontSize: 16 }} />
                      Keywords
                    </Typography>
                    <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
                      {firm.keywords.map((keyword, index) => (
                        <Chip
                          key={index}
                          label={keyword}
                          size="small"
                          variant="outlined"
                          color="primary"
                        />
                      ))}
                    </Box>
                  </Box>
                  
                  <Box sx={{ display: 'flex', alignItems: 'center', color: 'text.secondary' }}>
                    <CalendarIcon sx={{ mr: 1, fontSize: 16 }} />
                    <Typography variant="caption">
                      Added {formatDate(firm.created_at)}
                    </Typography>
                  </Box>
                </CardContent>
              </Card>
            </Grid>
          ))}
        </Grid>
      ) : (
        <Paper sx={{ p: 6, textAlign: 'center' }}>
          <BusinessIcon sx={{ fontSize: 64, color: 'text.secondary', mb: 2 }} />
          <Typography variant="h6" gutterBottom>
            No law firms yet
          </Typography>
          <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
            Add your first law firm to start organizing contract templates and comments.
          </Typography>
          <Button
            variant="contained"
            startIcon={<AddIcon />}
            onClick={() => setShowAddDialog(true)}
            size="large"
          >
            Add Your First Law Firm
          </Button>
        </Paper>
      )}

      {/* Add Law Firm Dialog */}
      <Dialog open={showAddDialog} onClose={() => setShowAddDialog(false)} maxWidth="sm" fullWidth>
        <DialogTitle>Add New Law Firm</DialogTitle>
        <DialogContent>
          <Box sx={{ pt: 2 }}>
            <TextField
              fullWidth
              label="Firm Name"
              value={formData.name}
              onChange={(e) => setFormData({ ...formData, name: e.target.value })}
              margin="normal"
              required
              placeholder="Enter law firm name"
            />
            <TextField
              fullWidth
              label="Keywords"
              value={formData.keywords}
              onChange={(e) => setFormData({ ...formData, keywords: e.target.value })}
              margin="normal"
              required
              multiline
              rows={3}
              placeholder="Enter keywords separated by commas (e.g., Smith & Associates, Smith Law, SLA)"
              helperText="These keywords will be used to automatically detect this law firm in contracts"
            />
          </Box>
        </DialogContent>
        <DialogActions sx={{ p: 3 }}>
          <Button
            onClick={() => {
              setShowAddDialog(false);
              setFormData({ name: '', keywords: '' });
            }}
          >
            Cancel
          </Button>
          <Button
            variant="contained"
            onClick={handleSubmit}
            disabled={addMutation.isLoading}
          >
            {addMutation.isLoading ? 'Adding...' : 'Add Law Firm'}
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
}

export default LawFirms; 