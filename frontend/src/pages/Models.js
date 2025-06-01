import React, { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import {
  Box,
  Typography,
  Paper,
  Button,
  Grid,
  Card,
  CardContent,
  CardActions,
  Chip,
  Alert,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  List,
  ListItem,
  ListItemText,
  ListItemSecondaryAction,
  IconButton,
  LinearProgress,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow
} from '@mui/material';
import {
  Download,
  Delete,
  Visibility,
  CloudDownload,
  Memory,
  Speed,
  ExpandMore,
  Info,
  PlayArrow,
  Stop
} from '@mui/icons-material';
import axios from 'axios';

const Models = () => {
  const [modelDetailsDialog, setModelDetailsDialog] = useState({ open: false, model: null });
  const [deleteConfirmDialog, setDeleteConfirmDialog] = useState({ open: false, model: null });
  const queryClient = useQueryClient();

  // Fetch trained models
  const { data: trainedModels = [], isLoading: trainedLoading } = useQuery({
    queryKey: ['trained-models'],
    queryFn: async () => {
      const response = await axios.get('http://localhost:8000/models');
      return response.data;
    }
  });

  // Fetch inference models
  const { data: inferenceModels = [], isLoading: inferenceLoading } = useQuery({
    queryKey: ['inference-models'],
    queryFn: async () => {
      const response = await axios.get('/inference/models');
      return response.data;
    }
  });

  // Delete model mutation
  const deleteModelMutation = useMutation({
    mutationFn: async (modelId) => {
      await axios.delete(`http://localhost:8000/models/${modelId}`);
    },
    onSuccess: () => {
      queryClient.invalidateQueries(['trained-models']);
      setDeleteConfirmDialog({ open: false, model: null });
    }
  });

  // Load model mutation
  const loadModelMutation = useMutation(
    async (modelId) => {
      await axios.post(`/inference/models/${modelId}/load`);
    },
    {
      onSuccess: () => {
        queryClient.invalidateQueries('available-models');
      },
    }
  );

  // Unload model mutation
  const unloadModelMutation = useMutation(
    async (modelId) => {
      await axios.post(`/inference/models/${modelId}/unload`);
    },
    {
      onSuccess: () => {
        queryClient.invalidateQueries('available-models');
      },
    }
  );

  const handleViewDetails = async (modelId) => {
    try {
      const response = await axios.get(`http://localhost:8000/models/${modelId}`);
      setModelDetailsDialog({ open: true, model: response.data });
    } catch (error) {
      console.error('Failed to load model details:', error);
    }
  };

  const handleDownloadModel = async (modelId, filename) => {
    try {
      const response = await axios.get(`http://localhost:8000/models/${modelId}/download`, {
        responseType: 'blob'
      });
      
      const url = window.URL.createObjectURL(new Blob([response.data]));
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', filename || `model_${modelId}.tar.gz`);
      document.body.appendChild(link);
      link.click();
      link.remove();
      window.URL.revokeObjectURL(url);
    } catch (error) {
      console.error('Failed to download model:', error);
    }
  };

  const formatFileSize = (bytes) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const formatDuration = (seconds) => {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = seconds % 60;
    return `${hours}h ${minutes}m ${secs}s`;
  };

  const getModelStatus = (model) => {
    const inferenceModel = inferenceModels.find(im => im.id === model.id);
    return inferenceModel?.loaded ? 'loaded' : 'available';
  };

  const getStatusColor = (status) => {
    switch (status) {
      case 'loaded':
        return 'success';
      case 'available':
        return 'default';
      case 'training':
        return 'warning';
      case 'failed':
        return 'error';
      default:
        return 'default';
    }
  };

  return (
    <Box sx={{ p: 3 }}>
      <Typography variant="h4" gutterBottom>
        Model Management
      </Typography>
      
      <Typography variant="body1" color="text.secondary" sx={{ mb: 3 }}>
        Manage your trained models, view details, and control inference loading.
      </Typography>

      {/* Trained Models */}
      <Paper sx={{ p: 3, mb: 3 }}>
        <Typography variant="h6" gutterBottom>
          Trained Models ({trainedModels.length})
        </Typography>

        {trainedLoading ? (
          <LinearProgress />
        ) : trainedModels.length === 0 ? (
          <Alert severity="info">
            No trained models yet. Complete a training job to see models here!
          </Alert>
        ) : (
          <Grid container spacing={2}>
            {trainedModels.map((model) => (
              <Grid item xs={12} md={6} lg={4} key={model.id}>
                <Card>
                  <CardContent>
                    <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                      <Typography variant="h6" sx={{ flexGrow: 1 }}>
                        {model.name}
                      </Typography>
                      <Chip
                        label={getModelStatus(model)}
                        color={getStatusColor(getModelStatus(model))}
                        size="small"
                      />
                    </Box>

                    <Typography variant="body2" color="text.secondary" gutterBottom>
                      Base: {model.base_model}
                    </Typography>

                    <Typography variant="body2" color="text.secondary" gutterBottom>
                      Created: {new Date(model.created_at).toLocaleDateString()}
                    </Typography>

                    {model.size && (
                      <Typography variant="body2" color="text.secondary" gutterBottom>
                        Size: {formatFileSize(model.size)}
                      </Typography>
                    )}

                    {model.training_duration && (
                      <Typography variant="body2" color="text.secondary" gutterBottom>
                        Training: {formatDuration(model.training_duration)}
                      </Typography>
                    )}

                    {model.metrics && (
                      <Box sx={{ mt: 1 }}>
                        <Typography variant="caption" color="text.secondary">
                          Final Loss: {model.metrics.final_loss?.toFixed(4)}
                        </Typography>
                      </Box>
                    )}
                  </CardContent>

                  <CardActions>
                    <Button
                      size="small"
                      startIcon={<Visibility />}
                      onClick={() => handleViewDetails(model.id)}
                    >
                      Details
                    </Button>

                    {getModelStatus(model) === 'available' ? (
                      <Button
                        size="small"
                        startIcon={<PlayArrow />}
                        onClick={() => loadModelMutation.mutate(model.id)}
                        disabled={loadModelMutation.isPending}
                      >
                        Load
                      </Button>
                    ) : (
                      <Button
                        size="small"
                        color="warning"
                        startIcon={<Stop />}
                        onClick={() => unloadModelMutation.mutate(model.id)}
                        disabled={unloadModelMutation.isPending}
                      >
                        Unload
                      </Button>
                    )}

                    <Button
                      size="small"
                      startIcon={<Download />}
                      onClick={() => handleDownloadModel(model.id, model.name)}
                    >
                      Export
                    </Button>

                    <IconButton
                      size="small"
                      color="error"
                      onClick={() => setDeleteConfirmDialog({ open: true, model })}
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

      {/* Inference Service Status */}
      <Paper sx={{ p: 3 }}>
        <Typography variant="h6" gutterBottom>
          Inference Service Status
        </Typography>

        {inferenceLoading ? (
          <LinearProgress />
        ) : (
          <TableContainer>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell>Model</TableCell>
                  <TableCell>Status</TableCell>
                  <TableCell>Memory Usage</TableCell>
                  <TableCell>Actions</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {inferenceModels.map((model) => (
                  <TableRow key={model.id}>
                    <TableCell>
                      <Typography variant="body2">{model.name}</Typography>
                      <Typography variant="caption" color="text.secondary">
                        {model.base_model}
                      </Typography>
                    </TableCell>
                    <TableCell>
                      <Chip
                        label={model.loaded ? 'Loaded' : 'Available'}
                        color={model.loaded ? 'success' : 'default'}
                        size="small"
                      />
                    </TableCell>
                    <TableCell>
                      {model.memory_usage ? (
                        <Box>
                          <Typography variant="body2">
                            {formatFileSize(model.memory_usage)}
                          </Typography>
                          {model.gpu_memory && (
                            <Typography variant="caption" color="text.secondary">
                              GPU: {formatFileSize(model.gpu_memory)}
                            </Typography>
                          )}
                        </Box>
                      ) : (
                        <Typography variant="body2" color="text.secondary">
                          Not loaded
                        </Typography>
                      )}
                    </TableCell>
                    <TableCell>
                      {model.loaded ? (
                        <Button
                          size="small"
                          color="warning"
                          startIcon={<Stop />}
                          onClick={() => unloadModelMutation.mutate(model.id)}
                          disabled={unloadModelMutation.isPending}
                        >
                          Unload
                        </Button>
                      ) : (
                        <Button
                          size="small"
                          startIcon={<PlayArrow />}
                          onClick={() => loadModelMutation.mutate(model.id)}
                          disabled={loadModelMutation.isPending}
                        >
                          Load
                        </Button>
                      )}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        )}

        {(loadModelMutation.isError || unloadModelMutation.isError) && (
          <Alert severity="error" sx={{ mt: 2 }}>
            {loadModelMutation.error?.message || unloadModelMutation.error?.message}
          </Alert>
        )}
      </Paper>

      {/* Model Details Dialog */}
      <Dialog
        open={modelDetailsDialog.open}
        onClose={() => setModelDetailsDialog({ open: false, model: null })}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>Model Details</DialogTitle>
        <DialogContent>
          {modelDetailsDialog.model && (
            <Box>
              <Grid container spacing={2} sx={{ mb: 3 }}>
                <Grid item xs={6}>
                  <Typography variant="subtitle2">Name:</Typography>
                  <Typography variant="body2">{modelDetailsDialog.model.name}</Typography>
                </Grid>
                <Grid item xs={6}>
                  <Typography variant="subtitle2">Base Model:</Typography>
                  <Typography variant="body2">{modelDetailsDialog.model.base_model}</Typography>
                </Grid>
                <Grid item xs={6}>
                  <Typography variant="subtitle2">Created:</Typography>
                  <Typography variant="body2">
                    {new Date(modelDetailsDialog.model.created_at).toLocaleString()}
                  </Typography>
                </Grid>
                <Grid item xs={6}>
                  <Typography variant="subtitle2">Size:</Typography>
                  <Typography variant="body2">
                    {formatFileSize(modelDetailsDialog.model.size || 0)}
                  </Typography>
                </Grid>
              </Grid>

              {/* Training Configuration */}
              {modelDetailsDialog.model.config && (
                <Accordion sx={{ mb: 2 }}>
                  <AccordionSummary expandIcon={<ExpandMore />}>
                    <Typography>Training Configuration</Typography>
                  </AccordionSummary>
                  <AccordionDetails>
                    <Grid container spacing={2}>
                      <Grid item xs={6}>
                        <Typography variant="caption">Learning Rate:</Typography>
                        <Typography variant="body2">
                          {modelDetailsDialog.model.config.learning_rate}
                        </Typography>
                      </Grid>
                      <Grid item xs={6}>
                        <Typography variant="caption">Batch Size:</Typography>
                        <Typography variant="body2">
                          {modelDetailsDialog.model.config.batch_size}
                        </Typography>
                      </Grid>
                      <Grid item xs={6}>
                        <Typography variant="caption">Epochs:</Typography>
                        <Typography variant="body2">
                          {modelDetailsDialog.model.config.num_epochs}
                        </Typography>
                      </Grid>
                      <Grid item xs={6}>
                        <Typography variant="caption">LoRA:</Typography>
                        <Typography variant="body2">
                          {modelDetailsDialog.model.config.use_lora ? 'Yes' : 'No'}
                        </Typography>
                      </Grid>
                    </Grid>
                  </AccordionDetails>
                </Accordion>
              )}

              {/* Training Metrics */}
              {modelDetailsDialog.model.metrics && (
                <Accordion sx={{ mb: 2 }}>
                  <AccordionSummary expandIcon={<ExpandMore />}>
                    <Typography>Training Metrics</Typography>
                  </AccordionSummary>
                  <AccordionDetails>
                    <Grid container spacing={2}>
                      <Grid item xs={6}>
                        <Typography variant="caption">Final Loss:</Typography>
                        <Typography variant="body2">
                          {modelDetailsDialog.model.metrics.final_loss?.toFixed(4)}
                        </Typography>
                      </Grid>
                      <Grid item xs={6}>
                        <Typography variant="caption">Best Loss:</Typography>
                        <Typography variant="body2">
                          {modelDetailsDialog.model.metrics.best_loss?.toFixed(4)}
                        </Typography>
                      </Grid>
                      <Grid item xs={6}>
                        <Typography variant="caption">Training Duration:</Typography>
                        <Typography variant="body2">
                          {formatDuration(modelDetailsDialog.model.training_duration || 0)}
                        </Typography>
                      </Grid>
                      <Grid item xs={6}>
                        <Typography variant="caption">Total Steps:</Typography>
                        <Typography variant="body2">
                          {modelDetailsDialog.model.metrics.total_steps}
                        </Typography>
                      </Grid>
                    </Grid>
                  </AccordionDetails>
                </Accordion>
              )}

              {/* Training Data */}
              {modelDetailsDialog.model.training_data && (
                <Accordion>
                  <AccordionSummary expandIcon={<ExpandMore />}>
                    <Typography>Training Data</Typography>
                  </AccordionSummary>
                  <AccordionDetails>
                    <List dense>
                      {modelDetailsDialog.model.training_data.map((pdf, index) => (
                        <ListItem key={index}>
                          <ListItemText
                            primary={pdf.filename}
                            secondary={`${pdf.chunks_count} chunks`}
                          />
                        </ListItem>
                      ))}
                    </List>
                  </AccordionDetails>
                </Accordion>
              )}
            </Box>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setModelDetailsDialog({ open: false, model: null })}>
            Close
          </Button>
        </DialogActions>
      </Dialog>

      {/* Delete Confirmation Dialog */}
      <Dialog
        open={deleteConfirmDialog.open}
        onClose={() => setDeleteConfirmDialog({ open: false, model: null })}
      >
        <DialogTitle>Delete Model</DialogTitle>
        <DialogContent>
          <Typography>
            Are you sure you want to delete the model "{deleteConfirmDialog.model?.name}"?
            This action cannot be undone.
          </Typography>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setDeleteConfirmDialog({ open: false, model: null })}>
            Cancel
          </Button>
          <Button
            color="error"
            onClick={() => deleteModelMutation.mutate(deleteConfirmDialog.model.id)}
            disabled={deleteModelMutation.isPending}
          >
            Delete
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default Models; 