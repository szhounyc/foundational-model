import React, { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import {
  Box,
  Typography,
  Paper,
  Button,
  LinearProgress,
  Alert,
  Grid,
  Card,
  CardContent,
  CardActions,
  Chip,
  IconButton,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  List,
  ListItem,
  ListItemText,
  ListItemSecondaryAction
} from '@mui/material';
import {
  CloudUpload,
  Description,
  Delete,
  Visibility,
  CheckCircle,
  Error,
  Pending
} from '@mui/icons-material';
import axios from 'axios';

const PDFUpload = () => {
  const [selectedFiles, setSelectedFiles] = useState([]);
  const [uploadProgress, setUploadProgress] = useState({});
  const [previewDialog, setPreviewDialog] = useState({ open: false, content: null });
  const queryClient = useQueryClient();

  // Fetch uploaded PDFs
  const { data: pdfs = [], isLoading } = useQuery({
    queryKey: ['pdfs'],
    queryFn: async () => {
      const response = await axios.get('http://localhost:8000/pdfs');
      return response.data;
    },
    refetchInterval: 5000
  });

  // Upload mutation
  const uploadMutation = useMutation({
    mutationFn: async (files) => {
      const results = [];
      for (const file of files) {
        const formData = new FormData();
        formData.append('file', file);
        
        try {
          const response = await axios.post('http://localhost:8000/pdfs/upload', formData, {
            headers: { 'Content-Type': 'multipart/form-data' },
            onUploadProgress: (progressEvent) => {
              const progress = Math.round((progressEvent.loaded * 100) / progressEvent.total);
              setUploadProgress(prev => ({ ...prev, [file.name]: progress }));
            }
          });
          results.push({ file: file.name, success: true, data: response.data });
        } catch (error) {
          results.push({ file: file.name, success: false, error: error.message });
        }
      }
      return results;
    },
    onSuccess: () => {
      queryClient.invalidateQueries(['pdfs']);
      setSelectedFiles([]);
      setUploadProgress({});
    }
  });

  // Delete mutation
  const deleteMutation = useMutation({
    mutationFn: async (pdfId) => {
      await axios.delete(`http://localhost:8000/pdfs/${pdfId}`);
    },
    onSuccess: () => {
      queryClient.invalidateQueries(['pdfs']);
    }
  });

  // Process mutation
  const processMutation = useMutation({
    mutationFn: async (pdfId) => {
      await axios.post(`http://localhost:8000/pdfs/${pdfId}/process`);
    },
    onSuccess: () => {
      queryClient.invalidateQueries(['pdfs']);
    }
  });

  const handleFileSelect = (event) => {
    const files = Array.from(event.target.files);
    setSelectedFiles(files);
  };

  const handleUpload = () => {
    if (selectedFiles.length > 0) {
      uploadMutation.mutate(selectedFiles);
    }
  };

  const handlePreview = async (pdfId) => {
    try {
      const response = await axios.get(`http://localhost:8000/pdfs/${pdfId}/preview`);
      setPreviewDialog({ open: true, content: response.data });
    } catch (error) {
      console.error('Failed to load preview:', error);
    }
  };

  const getStatusIcon = (status) => {
    switch (status) {
      case 'processed':
        return <CheckCircle color="success" />;
      case 'processing':
        return <Pending color="warning" />;
      case 'error':
        return <Error color="error" />;
      default:
        return <Description color="action" />;
    }
  };

  const getStatusColor = (status) => {
    switch (status) {
      case 'processed':
        return 'success';
      case 'processing':
        return 'warning';
      case 'error':
        return 'error';
      default:
        return 'default';
    }
  };

  return (
    <Box sx={{ p: 3 }}>
      <Typography variant="h4" gutterBottom>
        PDF Upload & Processing
      </Typography>
      
      <Typography variant="body1" color="text.secondary" sx={{ mb: 3 }}>
        Upload PDF files to extract text for model training. Files will be processed automatically.
      </Typography>

      {/* Upload Section */}
      <Paper sx={{ p: 3, mb: 3 }}>
        <Typography variant="h6" gutterBottom>
          Upload New PDFs
        </Typography>
        
        <Box sx={{ mb: 2 }}>
          <input
            accept=".pdf"
            style={{ display: 'none' }}
            id="pdf-upload"
            multiple
            type="file"
            onChange={handleFileSelect}
          />
          <label htmlFor="pdf-upload">
            <Button
              variant="outlined"
              component="span"
              startIcon={<CloudUpload />}
              sx={{ mr: 2 }}
            >
              Select PDF Files
            </Button>
          </label>
          
          <Button
            variant="contained"
            onClick={handleUpload}
            disabled={selectedFiles.length === 0 || uploadMutation.isPending}
            startIcon={<CloudUpload />}
          >
            Upload {selectedFiles.length > 0 && `(${selectedFiles.length} files)`}
          </Button>
        </Box>

        {selectedFiles.length > 0 && (
          <Box sx={{ mb: 2 }}>
            <Typography variant="subtitle2" gutterBottom>
              Selected Files:
            </Typography>
            <List dense>
              {selectedFiles.map((file, index) => (
                <ListItem key={index}>
                  <ListItemText
                    primary={file.name}
                    secondary={`${(file.size / 1024 / 1024).toFixed(2)} MB`}
                  />
                  {uploadProgress[file.name] && (
                    <Box sx={{ width: 100, ml: 2 }}>
                      <LinearProgress
                        variant="determinate"
                        value={uploadProgress[file.name]}
                      />
                      <Typography variant="caption">
                        {uploadProgress[file.name]}%
                      </Typography>
                    </Box>
                  )}
                </ListItem>
              ))}
            </List>
          </Box>
        )}

        {uploadMutation.isError && (
          <Alert severity="error" sx={{ mt: 2 }}>
            Upload failed: {uploadMutation.error?.message}
          </Alert>
        )}

        {uploadMutation.isSuccess && (
          <Alert severity="success" sx={{ mt: 2 }}>
            Files uploaded successfully!
          </Alert>
        )}
      </Paper>

      {/* Uploaded PDFs Section */}
      <Paper sx={{ p: 3 }}>
        <Typography variant="h6" gutterBottom>
          Uploaded PDFs ({pdfs.length})
        </Typography>

        {isLoading ? (
          <LinearProgress />
        ) : pdfs.length === 0 ? (
          <Alert severity="info">
            No PDFs uploaded yet. Upload some PDFs to get started!
          </Alert>
        ) : (
          <Grid container spacing={2}>
            {pdfs.map((pdf) => (
              <Grid item xs={12} md={6} lg={4} key={pdf.id}>
                <Card>
                  <CardContent>
                    <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                      {getStatusIcon(pdf.status)}
                      <Typography variant="h6" sx={{ ml: 1, flexGrow: 1 }}>
                        {pdf.filename}
                      </Typography>
                    </Box>
                    
                    <Typography variant="body2" color="text.secondary" gutterBottom>
                      Size: {(pdf.size / 1024 / 1024).toFixed(2)} MB
                    </Typography>
                    
                    <Typography variant="body2" color="text.secondary" gutterBottom>
                      Uploaded: {new Date(pdf.uploaded_at).toLocaleDateString()}
                    </Typography>
                    
                    <Chip
                      label={pdf.status}
                      color={getStatusColor(pdf.status)}
                      size="small"
                      sx={{ mb: 1 }}
                    />
                    
                    {pdf.chunks_count && (
                      <Typography variant="body2" color="text.secondary">
                        Text chunks: {pdf.chunks_count}
                      </Typography>
                    )}
                  </CardContent>
                  
                  <CardActions>
                    <Button
                      size="small"
                      startIcon={<Visibility />}
                      onClick={() => handlePreview(pdf.id)}
                    >
                      Preview
                    </Button>
                    
                    {pdf.status === 'uploaded' && (
                      <Button
                        size="small"
                        onClick={() => processMutation.mutate(pdf.id)}
                        disabled={processMutation.isPending}
                      >
                        Process
                      </Button>
                    )}
                    
                    <IconButton
                      size="small"
                      color="error"
                      onClick={() => deleteMutation.mutate(pdf.id)}
                      disabled={deleteMutation.isPending}
                    >
                      <Delete />
                    </IconButton>
                  </CardActions>
                </Card>
              </Grid>
            ))}
          </Grid>
        )}
      </Paper>

      {/* Preview Dialog */}
      <Dialog
        open={previewDialog.open}
        onClose={() => setPreviewDialog({ open: false, content: null })}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>PDF Preview</DialogTitle>
        <DialogContent>
          {previewDialog.content && (
            <Box>
              <Typography variant="subtitle2" gutterBottom>
                Extracted Text (first 1000 characters):
              </Typography>
              <Paper sx={{ p: 2, bgcolor: 'grey.50', maxHeight: 400, overflow: 'auto' }}>
                <Typography variant="body2" component="pre" sx={{ whiteSpace: 'pre-wrap' }}>
                  {previewDialog.content.text?.substring(0, 1000)}
                  {previewDialog.content.text?.length > 1000 && '...'}
                </Typography>
              </Paper>
              
              {previewDialog.content.chunks && (
                <Box sx={{ mt: 2 }}>
                  <Typography variant="subtitle2" gutterBottom>
                    Text Chunks: {previewDialog.content.chunks.length}
                  </Typography>
                  <List dense sx={{ maxHeight: 200, overflow: 'auto' }}>
                    {previewDialog.content.chunks.slice(0, 5).map((chunk, index) => (
                      <ListItem key={index}>
                        <ListItemText
                          primary={`Chunk ${index + 1}`}
                          secondary={chunk.substring(0, 100) + '...'}
                        />
                      </ListItem>
                    ))}
                  </List>
                </Box>
              )}
            </Box>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setPreviewDialog({ open: false, content: null })}>
            Close
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default PDFUpload; 