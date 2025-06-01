import React, { useState, useCallback } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Button,
  Grid,
  Alert,
  LinearProgress,
  Chip,
  List,
  ListItem,
  ListItemText,
  ListItemSecondaryAction,
  IconButton,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Paper,
  Divider,
} from '@mui/material';
import {
  CloudUpload as UploadIcon,
  Delete as DeleteIcon,
  Gavel as ReviewIcon,
  Description as PdfIcon,
  CheckCircle as CheckIcon,
  Error as ErrorIcon,
} from '@mui/icons-material';
import { useDropzone } from 'react-dropzone';
import { useMutation, useQuery } from 'react-query';
import axios from 'axios';

const API_BASE = '';

function ContractReview() {
  const [uploadedFiles, setUploadedFiles] = useState([]);
  const [selectedFiles, setSelectedFiles] = useState([]);
  const [reviewType, setReviewType] = useState('comprehensive');
  const [modelId, setModelId] = useState('sftj-s1xkr35z');
  const [reviewResult, setReviewResult] = useState(null);

  // Upload mutation
  const uploadMutation = useMutation(
    async (file) => {
      const formData = new FormData();
      formData.append('file', file);
      const response = await axios.post(`${API_BASE}/api/contracts/upload`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      return response.data;
    },
    {
      onSuccess: (data, file) => {
        setUploadedFiles(prev => [...prev, { ...data, file }]);
      },
    }
  );

  // Review mutation
  const reviewMutation = useMutation(
    async (reviewData) => {
      const response = await axios.post(`${API_BASE}/api/contracts/review`, reviewData);
      return response.data;
    },
    {
      onSuccess: (data) => {
        setReviewResult(data);
      },
    }
  );

  // Get available models
  const { data: modelsData } = useQuery(
    'inference-models',
    async () => {
      const response = await axios.get(`${API_BASE}/inference/models`);
      return response.data;
    },
    {
      refetchInterval: 30000,
    }
  );

  // Get uploaded files
  const { data: filesData } = useQuery(
    'uploaded-files',
    async () => {
      const response = await axios.get(`${API_BASE}/api/contracts/files`);
      return response.data;
    },
    {
      refetchInterval: 10000,
      onSuccess: (data) => {
        setUploadedFiles(data.files || []);
      },
    }
  );

  const onDrop = useCallback((acceptedFiles) => {
    acceptedFiles.forEach((file) => {
      if (file.type === 'application/pdf') {
        uploadMutation.mutate(file);
      }
    });
  }, [uploadMutation]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'application/pdf': ['.pdf'],
    },
    multiple: true,
  });

  const handleFileSelect = (fileId) => {
    setSelectedFiles(prev => {
      if (prev.includes(fileId)) {
        return prev.filter(id => id !== fileId);
      } else {
        return [...prev, fileId];
      }
    });
  };

  const handleRemoveFile = (fileId) => {
    setUploadedFiles(prev => prev.filter(file => file.id !== fileId));
    setSelectedFiles(prev => prev.filter(id => id !== fileId));
  };

  const handleReview = () => {
    if (selectedFiles.length === 0) {
      return;
    }

    reviewMutation.mutate({
      contract_files: selectedFiles,
      model_id: modelId,
      review_type: reviewType,
    });
  };

  const formatFileSize = (bytes) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  return (
    <Box sx={{ maxWidth: 1200, mx: 'auto' }}>
      <Typography variant="h4" gutterBottom sx={{ mb: 3, fontWeight: 600 }}>
        Contract Review
      </Typography>

      <Grid container spacing={3}>
        {/* Upload Section */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Upload Contract Files
              </Typography>
              
              <Paper
                {...getRootProps()}
                sx={{
                  p: 3,
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
                  {isDragActive ? 'Drop PDF files here' : 'Drag & drop PDF files here'}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  or click to select files
                </Typography>
                <Button
                  variant="contained"
                  startIcon={<UploadIcon />}
                  sx={{ mt: 2 }}
                >
                  Select Files
                </Button>
              </Paper>

              {uploadMutation.isLoading && (
                <Box sx={{ mt: 2 }}>
                  <LinearProgress />
                  <Typography variant="body2" sx={{ mt: 1 }}>
                    Uploading and processing files...
                  </Typography>
                </Box>
              )}

              {uploadMutation.isError && (
                <Alert severity="error" sx={{ mt: 2 }}>
                  Upload failed: {uploadMutation.error?.response?.data?.detail || 'Unknown error'}
                </Alert>
              )}
            </CardContent>
          </Card>
        </Grid>

        {/* Configuration Section */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Review Configuration
              </Typography>

              <FormControl fullWidth sx={{ mb: 2 }}>
                <InputLabel>Review Type</InputLabel>
                <Select
                  value={reviewType}
                  label="Review Type"
                  onChange={(e) => setReviewType(e.target.value)}
                >
                  <MenuItem value="comprehensive">Comprehensive Review</MenuItem>
                  <MenuItem value="risk_analysis">Risk Analysis</MenuItem>
                  <MenuItem value="quick">Quick Review</MenuItem>
                </Select>
              </FormControl>

              <FormControl fullWidth sx={{ mb: 3 }}>
                <InputLabel>AI Model</InputLabel>
                <Select
                  value={modelId}
                  label="AI Model"
                  onChange={(e) => setModelId(e.target.value)}
                >
                  {modelsData?.models?.map((model) => (
                    <MenuItem key={model.id} value={model.id}>
                      {model.name} ({model.type})
                    </MenuItem>
                  )) || [
                    <MenuItem key="default" value="sftj-s1xkr35z">
                      Fireworks Contract Model
                    </MenuItem>
                  ]}
                </Select>
              </FormControl>

              <Button
                variant="contained"
                size="large"
                fullWidth
                startIcon={<ReviewIcon />}
                onClick={handleReview}
                disabled={selectedFiles.length === 0 || reviewMutation.isLoading}
                sx={{ py: 1.5 }}
              >
                {reviewMutation.isLoading ? 'Reviewing...' : 'Start Review'}
              </Button>

              {reviewMutation.isLoading && (
                <Box sx={{ mt: 2 }}>
                  <LinearProgress />
                  <Typography variant="body2" sx={{ mt: 1 }}>
                    AI is analyzing your contracts...
                  </Typography>
                </Box>
              )}
            </CardContent>
          </Card>
        </Grid>

        {/* Uploaded Files */}
        {uploadedFiles.length > 0 && (
          <Grid item xs={12}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Uploaded Files ({uploadedFiles.length})
                </Typography>
                <List>
                  {uploadedFiles.map((file) => (
                    <ListItem
                      key={file.id}
                      sx={{
                        border: 1,
                        borderColor: selectedFiles.includes(file.id) ? 'primary.main' : 'grey.300',
                        borderRadius: 1,
                        mb: 1,
                        cursor: 'pointer',
                        backgroundColor: selectedFiles.includes(file.id) ? 'primary.light' : 'transparent',
                      }}
                      onClick={() => handleFileSelect(file.id)}
                    >
                      <PdfIcon color="error" sx={{ mr: 2 }} />
                      <ListItemText
                        primary={file.filename}
                        secondary={
                          <Box sx={{ display: 'flex', gap: 1, alignItems: 'center', mt: 0.5 }}>
                            <Chip
                              size="small"
                              icon={file.text_extracted ? <CheckIcon /> : <ErrorIcon />}
                              label={file.text_extracted ? 'Text Extracted' : 'Processing Failed'}
                              color={file.text_extracted ? 'success' : 'error'}
                            />
                            <Typography variant="caption">
                              {formatFileSize(file.file?.size || 0)}
                            </Typography>
                            {file.text_length && (
                              <Typography variant="caption">
                                • {file.text_length} characters
                              </Typography>
                            )}
                          </Box>
                        }
                      />
                      <ListItemSecondaryAction>
                        <IconButton
                          edge="end"
                          onClick={(e) => {
                            e.stopPropagation();
                            handleRemoveFile(file.id);
                          }}
                        >
                          <DeleteIcon />
                        </IconButton>
                      </ListItemSecondaryAction>
                    </ListItem>
                  ))}
                </List>
                {selectedFiles.length > 0 && (
                  <Alert severity="info" sx={{ mt: 2 }}>
                    {selectedFiles.length} file(s) selected for review
                  </Alert>
                )}
              </CardContent>
            </Card>
          </Grid>
        )}

        {/* Review Results */}
        {reviewResult && (
          <Grid item xs={12}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Review Results
                </Typography>
                
                <Box sx={{ mb: 2 }}>
                  <Chip
                    label={`Status: ${reviewResult.status}`}
                    color="success"
                    sx={{ mr: 1 }}
                  />
                  <Chip
                    label={`Processing Time: ${reviewResult.processing_time?.toFixed(1)}s`}
                    variant="outlined"
                  />
                </Box>

                <Divider sx={{ my: 2 }} />

                <Paper sx={{ p: 3, backgroundColor: 'grey.50' }}>
                  <Typography variant="body1" sx={{ whiteSpace: 'pre-wrap', lineHeight: 1.6 }}>
                    {reviewResult.review?.review || 'No review content available'}
                  </Typography>
                </Paper>

                {reviewResult.review && (
                  <Box sx={{ mt: 2, display: 'flex', gap: 1, flexWrap: 'wrap' }}>
                    <Chip
                      size="small"
                      label={`Model: ${reviewResult.review.model_id}`}
                      variant="outlined"
                    />
                    <Chip
                      size="small"
                      label={`Type: ${reviewResult.review.review_type}`}
                      variant="outlined"
                    />
                    <Chip
                      size="small"
                      label={`Sections: ${reviewResult.review.sections_reviewed}`}
                      variant="outlined"
                    />
                    <Chip
                      size="small"
                      label={`Tokens: ${reviewResult.review.tokens_generated}`}
                      variant="outlined"
                    />
                  </Box>
                )}
              </CardContent>
            </Card>
          </Grid>
        )}

        {reviewMutation.isError && (
          <Grid item xs={12}>
            <Alert severity="error">
              Review failed: {reviewMutation.error?.response?.data?.detail || 'Unknown error'}
            </Alert>
          </Grid>
        )}
      </Grid>
    </Box>
  );
}

export default ContractReview; 