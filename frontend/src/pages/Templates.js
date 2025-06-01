import React, { useState, useCallback } from 'react';
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
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Chip,
  Paper,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  ListItemSecondaryAction,
  IconButton,
  Divider,
} from '@mui/material';
import {
  Description as DocumentIcon,
  CloudUpload as UploadIcon,
  Business as BusinessIcon,
  Category as CategoryIcon,
  CalendarToday as CalendarIcon,
  Delete as DeleteIcon,
  Download as DownloadIcon,
  Visibility as ViewIcon,
} from '@mui/icons-material';
import { useDropzone } from 'react-dropzone';
import { useQuery, useMutation, useQueryClient } from 'react-query';
import axios from 'axios';

const API_BASE = process.env.REACT_APP_API_URL || '';

const TEMPLATE_CATEGORIES = [
  { label: 'Purchase Agreement', value: 'purchase_agreement' },
  { label: 'Rider', value: 'rider' }, 
  { label: 'Legal Comments', value: 'legal_comments' }
];

function Templates() {
  const [showUploadDialog, setShowUploadDialog] = useState(false);
  const [uploadFile, setUploadFile] = useState(null);
  const [formData, setFormData] = useState({
    law_firm_id: '',
    template_type: '',
    description: ''
  });
  const [alert, setAlert] = useState(null);
  const queryClient = useQueryClient();

  // Fetch templates
  const { data: templatesData, isLoading } = useQuery(
    'templates',
    async () => {
      const response = await axios.get(`${API_BASE}/api/templates`);
      return response.data;
    },
    {
      refetchInterval: 30000,
    }
  );

  // Fetch law firms for selection
  const { data: lawFirmsData } = useQuery(
    'law-firms',
    async () => {
      const response = await axios.get(`${API_BASE}/api/law-firms`);
      return response.data;
    }
  );

  // Upload template mutation
  const uploadMutation = useMutation(
    async (templateData) => {
      const formData = new FormData();
      formData.append('file', templateData.file);
      
      const response = await axios.post(
        `${API_BASE}/api/templates/upload?law_firm_id=${templateData.law_firm_id}&template_type=${templateData.template_type}`, 
        formData,
        {
          headers: { 'Content-Type': 'multipart/form-data' },
        }
      );
      return response.data;
    },
    {
      onSuccess: () => {
        queryClient.invalidateQueries('templates');
        setShowUploadDialog(false);
        setUploadFile(null);
        setFormData({ law_firm_id: '', template_type: '', description: '' });
        setAlert({ type: 'success', message: 'Template uploaded successfully!' });
        setTimeout(() => setAlert(null), 3000);
      },
      onError: (error) => {
        setAlert({ type: 'error', message: error.response?.data?.detail || 'Failed to upload template' });
        setTimeout(() => setAlert(null), 5000);
      }
    }
  );

  // Delete template mutation
  const deleteMutation = useMutation(
    async (templateId) => {
      const response = await axios.delete(`${API_BASE}/api/templates/${templateId}`);
      return response.data;
    },
    {
      onSuccess: () => {
        queryClient.invalidateQueries('templates');
        setAlert({ type: 'success', message: 'Template deleted successfully!' });
        setTimeout(() => setAlert(null), 3000);
      },
      onError: (error) => {
        setAlert({ type: 'error', message: error.response?.data?.detail || 'Failed to delete template' });
        setTimeout(() => setAlert(null), 5000);
      }
    }
  );

  const onDrop = useCallback((acceptedFiles) => {
    if (acceptedFiles.length > 0) {
      const file = acceptedFiles[0];
      if (file.type === 'application/pdf' || file.type === 'application/vnd.openxmlformats-officedocument.wordprocessingml.document') {
        setUploadFile(file);
      } else {
        setAlert({ type: 'error', message: 'Please select a PDF or DOCX file' });
        setTimeout(() => setAlert(null), 3000);
      }
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'application/pdf': ['.pdf'],
      'application/vnd.openxmlformats-officedocument.wordprocessingml.document': ['.docx'],
    },
    multiple: false,
  });

  const handleUpload = () => {
    if (!uploadFile || !formData.law_firm_id || !formData.template_type) {
      setAlert({ type: 'error', message: 'Please fill in all required fields and select a file' });
      setTimeout(() => setAlert(null), 3000);
      return;
    }

    uploadMutation.mutate({
      file: uploadFile,
      ...formData
    });
  };

  // Handle template view
  const handleViewTemplate = async (templateId) => {
    try {
      const response = await axios.get(`${API_BASE}/api/templates/${templateId}/content`);
      // Open content in a new window or modal
      const newWindow = window.open('', '_blank');
      newWindow.document.write(`
        <html>
          <head><title>Template Content</title></head>
          <body style="font-family: Arial, sans-serif; padding: 20px;">
            <h2>Template Content</h2>
            <pre style="white-space: pre-wrap; word-wrap: break-word;">${response.data.content}</pre>
          </body>
        </html>
      `);
    } catch (error) {
      setAlert({ type: 'error', message: 'Failed to load template content' });
      setTimeout(() => setAlert(null), 3000);
    }
  };

  // Handle template download
  const handleDownloadTemplate = async (templateId, filename) => {
    try {
      const response = await axios.get(`${API_BASE}/api/templates/${templateId}/download`, {
        responseType: 'blob'
      });
      
      const url = window.URL.createObjectURL(new Blob([response.data]));
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', filename);
      document.body.appendChild(link);
      link.click();
      link.remove();
      window.URL.revokeObjectURL(url);
    } catch (error) {
      setAlert({ type: 'error', message: 'Failed to download template' });
      setTimeout(() => setAlert(null), 3000);
    }
  };

  // Handle template delete
  const handleDeleteTemplate = (templateId) => {
    if (window.confirm('Are you sure you want to delete this template?')) {
      deleteMutation.mutate(templateId);
    }
  };

  const templates = templatesData?.templates || [];
  const lawFirms = lawFirmsData?.law_firms || [];

  const formatFileSize = (bytes) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const formatDate = (dateString) => {
    return new Date(dateString).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    });
  };

  const getLawFirmName = (lawFirmId) => {
    const firm = lawFirms.find(f => f.id === lawFirmId);
    return firm ? firm.name : 'Unknown';
  };

  if (isLoading) {
    return (
      <Box sx={{ maxWidth: 1200, mx: 'auto', textAlign: 'center', py: 8 }}>
        <Typography variant="h6">Loading templates...</Typography>
      </Box>
    );
  }

  return (
    <Box sx={{ maxWidth: 1200, mx: 'auto' }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Box>
          <Typography variant="h4" gutterBottom sx={{ fontWeight: 600 }}>
            Contract Templates Library
          </Typography>
          <Typography variant="body1" color="text.secondary">
            Manage contract templates organized by law firm and category
          </Typography>
        </Box>
        <Button
          variant="contained"
          startIcon={<UploadIcon />}
          onClick={() => setShowUploadDialog(true)}
          size="large"
        >
          Upload Template
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
                  <DocumentIcon />
                </Box>
                <Typography variant="h4" sx={{ fontWeight: 600 }}>
                  {templates.length}
                </Typography>
              </Box>
              <Typography variant="body2" color="text.secondary">
                Total Templates
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
                    backgroundColor: 'success.light',
                    color: 'success.main',
                    mr: 2,
                  }}
                >
                  <CategoryIcon />
                </Box>
                <Typography variant="h4" sx={{ fontWeight: 600 }}>
                  {new Set(templates.map(t => t.template_type)).size}
                </Typography>
              </Box>
              <Typography variant="body2" color="text.secondary">
                Categories
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
                  <BusinessIcon />
                </Box>
                <Typography variant="h4" sx={{ fontWeight: 600 }}>
                  {new Set(templates.map(t => t.law_firm_id)).size}
                </Typography>
              </Box>
              <Typography variant="body2" color="text.secondary">
                Law Firms
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Templates List */}
      {templates.length > 0 ? (
        <Grid container spacing={3}>
          {templates.map((template) => (
            <Grid item xs={12} md={6} lg={4} key={template.id}>
              <Card sx={{ height: '100%' }}>
                <CardContent>
                  <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                    <DocumentIcon sx={{ color: 'primary.main', mr: 1 }} />
                    <Typography variant="h6" sx={{ flexGrow: 1 }} noWrap>
                      {template.filename}
                    </Typography>
                    <IconButton size="small" color="primary" onClick={() => handleViewTemplate(template.id)}>
                      <ViewIcon />
                    </IconButton>
                    <IconButton size="small" color="info" onClick={() => handleDownloadTemplate(template.id, template.filename)}>
                      <DownloadIcon />
                    </IconButton>
                    <IconButton size="small" color="error" onClick={() => handleDeleteTemplate(template.id)}>
                      <DeleteIcon />
                    </IconButton>
                  </Box>
                  
                  <Divider sx={{ mb: 2 }} />
                  
                  <Box sx={{ mb: 2 }}>
                    <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                      <BusinessIcon sx={{ mr: 1, fontSize: 16, color: 'text.secondary' }} />
                      <Typography variant="body2" color="text.secondary">
                        {getLawFirmName(template.law_firm_id)}
                      </Typography>
                    </Box>
                    
                    <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                      <CategoryIcon sx={{ mr: 1, fontSize: 16, color: 'text.secondary' }} />
                      <Chip
                        label={template.template_type}
                        size="small"
                        variant="outlined"
                        color="primary"
                      />
                    </Box>
                    
                    {template.description && (
                      <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                        {template.description}
                      </Typography>
                    )}
                    
                    <Typography variant="caption" color="text.secondary">
                      Size: {formatFileSize(template.file_size)}
                    </Typography>
                  </Box>
                  
                  <Box sx={{ display: 'flex', alignItems: 'center', color: 'text.secondary' }}>
                    <CalendarIcon sx={{ mr: 1, fontSize: 16 }} />
                    <Typography variant="caption">
                      Uploaded {formatDate(template.created_at)}
                    </Typography>
                  </Box>
                </CardContent>
              </Card>
            </Grid>
          ))}
        </Grid>
      ) : (
        <Paper sx={{ p: 6, textAlign: 'center' }}>
          <DocumentIcon sx={{ fontSize: 64, color: 'text.secondary', mb: 2 }} />
          <Typography variant="h6" gutterBottom>
            No templates yet
          </Typography>
          <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
            Upload your first contract template to start building your library.
          </Typography>
          <Button
            variant="contained"
            startIcon={<UploadIcon />}
            onClick={() => setShowUploadDialog(true)}
            size="large"
          >
            Upload Your First Template
          </Button>
        </Paper>
      )}

      {/* Upload Template Dialog */}
      <Dialog open={showUploadDialog} onClose={() => setShowUploadDialog(false)} maxWidth="md" fullWidth>
        <DialogTitle>Upload Contract Template</DialogTitle>
        <DialogContent>
          <Grid container spacing={3} sx={{ pt: 2 }}>
            {/* File Upload Area */}
            <Grid item xs={12}>
              <Paper
                {...getRootProps()}
                sx={{
                  p: 4,
                  border: '2px dashed',
                  borderColor: isDragActive ? 'primary.main' : 'grey.300',
                  backgroundColor: isDragActive ? 'primary.light' : 'grey.50',
                  cursor: 'pointer',
                  textAlign: 'center',
                  transition: 'all 0.3s ease',
                  '&:hover': {
                    borderColor: 'primary.main',
                    backgroundColor: 'primary.light',
                  },
                }}
              >
                <input {...getInputProps()} />
                <UploadIcon sx={{ fontSize: 48, color: 'primary.main', mb: 2 }} />
                <Typography variant="h6" gutterBottom>
                  {isDragActive ? 'Drop PDF or DOCX file here' : 'Drag & drop PDF or DOCX file here'}
                </Typography>
                <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                  or click to select file
                </Typography>
                {uploadFile && (
                  <Chip
                    label={`Selected: ${uploadFile.name}`}
                    color="primary"
                    variant="outlined"
                    sx={{ mt: 1 }}
                  />
                )}
              </Paper>
            </Grid>

            {/* Form Fields */}
            <Grid item xs={12} sm={6}>
              <FormControl fullWidth required>
                <InputLabel>Law Firm</InputLabel>
                <Select
                  value={formData.law_firm_id}
                  label="Law Firm"
                  onChange={(e) => setFormData({ ...formData, law_firm_id: e.target.value })}
                >
                  {lawFirms.map((firm) => (
                    <MenuItem key={firm.id} value={firm.id}>
                      {firm.name}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Grid>

            <Grid item xs={12} sm={6}>
              <FormControl fullWidth required>
                <InputLabel>Template Type</InputLabel>
                <Select
                  value={formData.template_type}
                  label="Template Type"
                  onChange={(e) => setFormData(prev => ({ ...prev, template_type: e.target.value }))}
                >
                  {TEMPLATE_CATEGORIES.map((category) => (
                    <MenuItem key={category.value} value={category.value}>
                      {category.label}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Grid>

            <Grid item xs={12}>
              <TextField
                fullWidth
                label="Description (Optional)"
                value={formData.description}
                onChange={(e) => setFormData({ ...formData, description: e.target.value })}
                multiline
                rows={3}
                placeholder="Enter a brief description of this template..."
              />
            </Grid>
          </Grid>
        </DialogContent>
        <DialogActions sx={{ p: 3 }}>
          <Button
            onClick={() => {
              setShowUploadDialog(false);
              setUploadFile(null);
              setFormData({ law_firm_id: '', template_type: '', description: '' });
            }}
          >
            Cancel
          </Button>
          <Button
            variant="contained"
            onClick={handleUpload}
            disabled={uploadMutation.isLoading || !uploadFile}
          >
            {uploadMutation.isLoading ? 'Uploading...' : 'Upload Template'}
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
}

export default Templates; 