import React, { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import {
  Box,
  Typography,
  Paper,
  Button,
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Grid,
  Card,
  CardContent,
  CardActions,
  LinearProgress,
  Alert,
  Chip,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  List,
  ListItem,
  ListItemText,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Slider,
  FormControlLabel,
  Switch
} from '@mui/material';
import {
  PlayArrow,
  Stop,
  Visibility,
  ExpandMore,
  Settings,
  Memory,
  Speed,
  TrendingUp
} from '@mui/icons-material';
import axios from 'axios';

const ModelTraining = () => {
  const [trainingConfig, setTrainingConfig] = useState({
    model_name: 'microsoft/DialoGPT-small',
    learning_rate: 5e-5,
    batch_size: 4,
    num_epochs: 3,
    max_length: 512,
    warmup_steps: 100,
    save_steps: 500,
    eval_steps: 500,
    gradient_accumulation_steps: 1,
    fp16: true,
    use_lora: true,
    lora_r: 16,
    lora_alpha: 32,
    lora_dropout: 0.1
  });
  const [selectedPdfs, setSelectedPdfs] = useState([]);
  const [jobDetailsDialog, setJobDetailsDialog] = useState({ open: false, job: null });
  const queryClient = useQueryClient();

  // Available models
  const availableModels = [
    { value: 'microsoft/DialoGPT-small', label: 'DialoGPT Small (117M params)' },
    { value: 'microsoft/DialoGPT-medium', label: 'DialoGPT Medium (345M params)' },
    { value: 'deepseek-ai/deepseek-coder-1.3b-base', label: 'DeepSeek Coder 1.3B' },
    { value: 'microsoft/CodeGPT-small-py', label: 'CodeGPT Small Python' },
    { value: 'distilgpt2', label: 'DistilGPT2 (82M params)' },
    { value: 'gpt2', label: 'GPT2 (124M params)' }
  ];

  // Fetch processed PDFs
  const { data: pdfs = [] } = useQuery({
    queryKey: ['processed-pdfs'],
    queryFn: async () => {
      const response = await axios.get('http://localhost:8000/pdfs?status=processed');
      return response.data;
    }
  });

  // Fetch training jobs
  const { data: jobs = [], isLoading: jobsLoading } = useQuery({
    queryKey: ['training-jobs'],
    queryFn: async () => {
      const response = await axios.get('http://localhost:8000/training/jobs');
      return response.data;
    },
    refetchInterval: 2000
  });

  // Start training mutation
  const startTrainingMutation = useMutation({
    mutationFn: async (config) => {
      const response = await axios.post('http://localhost:8000/training/start', config);
      return response.data;
    },
    onSuccess: () => {
      queryClient.invalidateQueries(['training-jobs']);
    }
  });

  // Stop training mutation
  const stopTrainingMutation = useMutation({
    mutationFn: async (jobId) => {
      await axios.post(`http://localhost:8000/training/jobs/${jobId}/stop`);
    },
    onSuccess: () => {
      queryClient.invalidateQueries(['training-jobs']);
    }
  });

  const handleConfigChange = (field, value) => {
    setTrainingConfig(prev => ({ ...prev, [field]: value }));
  };

  const handleStartTraining = () => {
    const config = {
      ...trainingConfig,
      pdf_ids: selectedPdfs
    };
    startTrainingMutation.mutate(config);
  };

  const handleViewJobDetails = async (jobId) => {
    try {
      const response = await axios.get(`http://localhost:8000/training/jobs/${jobId}`);
      setJobDetailsDialog({ open: true, job: response.data });
    } catch (error) {
      console.error('Failed to load job details:', error);
    }
  };

  const getStatusColor = (status) => {
    switch (status) {
      case 'running':
        return 'primary';
      case 'completed':
        return 'success';
      case 'failed':
        return 'error';
      case 'stopped':
        return 'warning';
      default:
        return 'default';
    }
  };

  const formatDuration = (seconds) => {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = seconds % 60;
    return `${hours}h ${minutes}m ${secs}s`;
  };

  return (
    <Box sx={{ p: 3 }}>
      <Typography variant="h4" gutterBottom>
        Model Training
      </Typography>
      
      <Typography variant="body1" color="text.secondary" sx={{ mb: 3 }}>
        Fine-tune language models using your processed PDF data. Configure training parameters and monitor progress.
      </Typography>

      <Grid container spacing={3}>
        {/* Training Configuration */}
        <Grid item xs={12} lg={6}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Training Configuration
            </Typography>

            {/* Model Selection */}
            <FormControl fullWidth sx={{ mb: 2 }}>
              <InputLabel>Base Model</InputLabel>
              <Select
                value={trainingConfig.model_name}
                onChange={(e) => handleConfigChange('model_name', e.target.value)}
                label="Base Model"
              >
                {availableModels.map((model) => (
                  <MenuItem key={model.value} value={model.value}>
                    {model.label}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>

            {/* Data Selection */}
            <FormControl fullWidth sx={{ mb: 2 }}>
              <InputLabel>Training Data (PDFs)</InputLabel>
              <Select
                multiple
                value={selectedPdfs}
                onChange={(e) => setSelectedPdfs(e.target.value)}
                label="Training Data (PDFs)"
                renderValue={(selected) => (
                  <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                    {selected.map((value) => {
                      const pdf = pdfs.find(p => p.id === value);
                      return (
                        <Chip key={value} label={pdf?.filename || value} size="small" />
                      );
                    })}
                  </Box>
                )}
              >
                {pdfs.map((pdf) => (
                  <MenuItem key={pdf.id} value={pdf.id}>
                    {pdf.filename} ({pdf.chunks_count} chunks)
                  </MenuItem>
                ))}
              </Select>
            </FormControl>

            {/* Basic Parameters */}
            <Accordion sx={{ mb: 2 }}>
              <AccordionSummary expandIcon={<ExpandMore />}>
                <Typography>Basic Parameters</Typography>
              </AccordionSummary>
              <AccordionDetails>
                <Grid container spacing={2}>
                  <Grid item xs={6}>
                    <TextField
                      fullWidth
                      label="Learning Rate"
                      type="number"
                      value={trainingConfig.learning_rate}
                      onChange={(e) => handleConfigChange('learning_rate', parseFloat(e.target.value))}
                      inputProps={{ step: 0.00001, min: 0.00001, max: 0.01 }}
                    />
                  </Grid>
                  <Grid item xs={6}>
                    <TextField
                      fullWidth
                      label="Batch Size"
                      type="number"
                      value={trainingConfig.batch_size}
                      onChange={(e) => handleConfigChange('batch_size', parseInt(e.target.value))}
                      inputProps={{ min: 1, max: 32 }}
                    />
                  </Grid>
                  <Grid item xs={6}>
                    <TextField
                      fullWidth
                      label="Epochs"
                      type="number"
                      value={trainingConfig.num_epochs}
                      onChange={(e) => handleConfigChange('num_epochs', parseInt(e.target.value))}
                      inputProps={{ min: 1, max: 10 }}
                    />
                  </Grid>
                  <Grid item xs={6}>
                    <TextField
                      fullWidth
                      label="Max Length"
                      type="number"
                      value={trainingConfig.max_length}
                      onChange={(e) => handleConfigChange('max_length', parseInt(e.target.value))}
                      inputProps={{ min: 128, max: 2048 }}
                    />
                  </Grid>
                </Grid>
              </AccordionDetails>
            </Accordion>

            {/* Advanced Parameters */}
            <Accordion sx={{ mb: 2 }}>
              <AccordionSummary expandIcon={<ExpandMore />}>
                <Typography>Advanced Parameters</Typography>
              </AccordionSummary>
              <AccordionDetails>
                <Grid container spacing={2}>
                  <Grid item xs={12}>
                    <FormControlLabel
                      control={
                        <Switch
                          checked={trainingConfig.fp16}
                          onChange={(e) => handleConfigChange('fp16', e.target.checked)}
                        />
                      }
                      label="Mixed Precision (FP16)"
                    />
                  </Grid>
                  <Grid item xs={12}>
                    <FormControlLabel
                      control={
                        <Switch
                          checked={trainingConfig.use_lora}
                          onChange={(e) => handleConfigChange('use_lora', e.target.checked)}
                        />
                      }
                      label="Use LoRA (Low-Rank Adaptation)"
                    />
                  </Grid>
                  {trainingConfig.use_lora && (
                    <>
                      <Grid item xs={4}>
                        <TextField
                          fullWidth
                          label="LoRA Rank"
                          type="number"
                          value={trainingConfig.lora_r}
                          onChange={(e) => handleConfigChange('lora_r', parseInt(e.target.value))}
                          inputProps={{ min: 1, max: 64 }}
                        />
                      </Grid>
                      <Grid item xs={4}>
                        <TextField
                          fullWidth
                          label="LoRA Alpha"
                          type="number"
                          value={trainingConfig.lora_alpha}
                          onChange={(e) => handleConfigChange('lora_alpha', parseInt(e.target.value))}
                          inputProps={{ min: 1, max: 128 }}
                        />
                      </Grid>
                      <Grid item xs={4}>
                        <TextField
                          fullWidth
                          label="LoRA Dropout"
                          type="number"
                          value={trainingConfig.lora_dropout}
                          onChange={(e) => handleConfigChange('lora_dropout', parseFloat(e.target.value))}
                          inputProps={{ step: 0.1, min: 0, max: 1 }}
                        />
                      </Grid>
                    </>
                  )}
                </Grid>
              </AccordionDetails>
            </Accordion>

            <Button
              variant="contained"
              size="large"
              startIcon={<PlayArrow />}
              onClick={handleStartTraining}
              disabled={selectedPdfs.length === 0 || startTrainingMutation.isPending}
              fullWidth
            >
              Start Training
            </Button>

            {startTrainingMutation.isError && (
              <Alert severity="error" sx={{ mt: 2 }}>
                Failed to start training: {startTrainingMutation.error?.message}
              </Alert>
            )}
          </Paper>
        </Grid>

        {/* Training Jobs */}
        <Grid item xs={12} lg={6}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Training Jobs
            </Typography>

            {jobsLoading ? (
              <LinearProgress />
            ) : jobs.length === 0 ? (
              <Alert severity="info">
                No training jobs yet. Start your first training job!
              </Alert>
            ) : (
              <List>
                {jobs.map((job) => (
                  <Card key={job.id} sx={{ mb: 2 }}>
                    <CardContent>
                      <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                        <Typography variant="h6" sx={{ flexGrow: 1 }}>
                          {job.model_name}
                        </Typography>
                        <Chip
                          label={job.status}
                          color={getStatusColor(job.status)}
                          size="small"
                        />
                      </Box>

                      <Typography variant="body2" color="text.secondary" gutterBottom>
                        Started: {new Date(job.created_at).toLocaleString()}
                      </Typography>

                      {job.status === 'running' && job.progress && (
                        <Box sx={{ mt: 2 }}>
                          <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                            <Typography variant="body2">
                              Epoch {job.progress.current_epoch}/{job.progress.total_epochs}
                            </Typography>
                            <Typography variant="body2">
                              {Math.round(job.progress.progress * 100)}%
                            </Typography>
                          </Box>
                          <LinearProgress
                            variant="determinate"
                            value={job.progress.progress * 100}
                          />
                          {job.progress.loss && (
                            <Typography variant="body2" sx={{ mt: 1 }}>
                              Loss: {job.progress.loss.toFixed(4)}
                            </Typography>
                          )}
                        </Box>
                      )}

                      {job.duration && (
                        <Typography variant="body2" color="text.secondary">
                          Duration: {formatDuration(job.duration)}
                        </Typography>
                      )}
                    </CardContent>

                    <CardActions>
                      <Button
                        size="small"
                        startIcon={<Visibility />}
                        onClick={() => handleViewJobDetails(job.id)}
                      >
                        Details
                      </Button>
                      
                      {job.status === 'running' && (
                        <Button
                          size="small"
                          color="error"
                          startIcon={<Stop />}
                          onClick={() => stopTrainingMutation.mutate(job.id)}
                          disabled={stopTrainingMutation.isPending}
                        >
                          Stop
                        </Button>
                      )}
                    </CardActions>
                  </Card>
                ))}
              </List>
            )}
          </Paper>
        </Grid>
      </Grid>

      {/* Job Details Dialog */}
      <Dialog
        open={jobDetailsDialog.open}
        onClose={() => setJobDetailsDialog({ open: false, job: null })}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>Training Job Details</DialogTitle>
        <DialogContent>
          {jobDetailsDialog.job && (
            <Box>
              <Grid container spacing={2}>
                <Grid item xs={6}>
                  <Typography variant="subtitle2">Model:</Typography>
                  <Typography variant="body2">{jobDetailsDialog.job.model_name}</Typography>
                </Grid>
                <Grid item xs={6}>
                  <Typography variant="subtitle2">Status:</Typography>
                  <Chip
                    label={jobDetailsDialog.job.status}
                    color={getStatusColor(jobDetailsDialog.job.status)}
                    size="small"
                  />
                </Grid>
                <Grid item xs={6}>
                  <Typography variant="subtitle2">Learning Rate:</Typography>
                  <Typography variant="body2">{jobDetailsDialog.job.config?.learning_rate}</Typography>
                </Grid>
                <Grid item xs={6}>
                  <Typography variant="subtitle2">Batch Size:</Typography>
                  <Typography variant="body2">{jobDetailsDialog.job.config?.batch_size}</Typography>
                </Grid>
                <Grid item xs={6}>
                  <Typography variant="subtitle2">Epochs:</Typography>
                  <Typography variant="body2">{jobDetailsDialog.job.config?.num_epochs}</Typography>
                </Grid>
                <Grid item xs={6}>
                  <Typography variant="subtitle2">Use LoRA:</Typography>
                  <Typography variant="body2">{jobDetailsDialog.job.config?.use_lora ? 'Yes' : 'No'}</Typography>
                </Grid>
              </Grid>

              {jobDetailsDialog.job.logs && (
                <Box sx={{ mt: 3 }}>
                  <Typography variant="subtitle2" gutterBottom>
                    Training Logs:
                  </Typography>
                  <Paper sx={{ p: 2, bgcolor: 'grey.50', maxHeight: 300, overflow: 'auto' }}>
                    <Typography variant="body2" component="pre" sx={{ whiteSpace: 'pre-wrap' }}>
                      {jobDetailsDialog.job.logs}
                    </Typography>
                  </Paper>
                </Box>
              )}
            </Box>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setJobDetailsDialog({ open: false, job: null })}>
            Close
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default ModelTraining; 