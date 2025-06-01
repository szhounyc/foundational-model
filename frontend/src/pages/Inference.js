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
  Alert,
  Chip,
  Slider,
  FormControlLabel,
  Switch,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  List,
  ListItem,
  ListItemText,
  ListItemSecondaryAction,
  IconButton,
  Divider
} from '@mui/material';
import {
  Send,
  Clear,
  Download,
  Settings,
  ExpandMore,
  PlayArrow,
  Stop,
  ContentCopy
} from '@mui/icons-material';
import axios from 'axios';

const Inference = () => {
  const [selectedModel, setSelectedModel] = useState('');
  const [prompt, setPrompt] = useState('');
  const [generationConfig, setGenerationConfig] = useState({
    max_length: 100,
    temperature: 0.7,
    top_p: 0.9,
    top_k: 50,
    repetition_penalty: 1.1,
    do_sample: true,
    num_return_sequences: 1
  });
  const [conversationHistory, setConversationHistory] = useState([]);
  const [batchPrompts, setBatchPrompts] = useState(['']);

  const queryClient = useQueryClient();

  // Get available models
  const { data: modelsData, refetch } = useQuery(
    'available-models',
    async () => {
      const response = await axios.get('/inference/models');
      return response.data;
    },
    {
      refetchInterval: 5000,
    }
  );

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

  // Generate text mutation
  const generateMutation = useMutation(
    async ({ prompt, modelId, maxTokens, temperature }) => {
      const response = await axios.post('/inference/generate', {
        prompt,
        model_id: modelId,
        max_tokens: maxTokens,
        temperature,
      });
      return response.data;
    },
    {
      onSuccess: (data) => {
        setConversationHistory(prev => [
          ...prev,
          {
            type: 'user',
            content: prompt,
            timestamp: new Date()
          },
          {
            type: 'assistant',
            content: data.generated_text,
            timestamp: new Date(),
            metadata: {
              generation_time: data.generation_time,
              tokens_generated: data.tokens_generated
            }
          }
        ]);
        setPrompt('');
      }
    }
  );

  // Batch generate mutation
  const batchGenerateMutation = useMutation(
    async ({ prompts, modelId, maxTokens, temperature }) => {
      const response = await axios.post('/inference/batch_generate', {
        prompts,
        model_id: modelId,
        max_tokens: maxTokens,
        temperature,
      });
      return response.data;
    },
    {
      onSuccess: (data) => {
        setBatchResults(data);
      },
    }
  );

  const handleGenerate = () => {
    if (prompt.trim() && selectedModel) {
      generateMutation.mutate({
        prompt: prompt.trim(),
        modelId: selectedModel,
        maxTokens: generationConfig.max_length,
        temperature: generationConfig.temperature
      });
    }
  };

  const handleBatchGenerate = () => {
    const validPrompts = batchPrompts.filter(p => p.trim());
    if (validPrompts.length > 0 && selectedModel) {
      batchGenerateMutation.mutate({
        prompts: validPrompts,
        modelId: selectedModel,
        maxTokens: generationConfig.max_length,
        temperature: generationConfig.temperature
      });
    }
  };

  const handleConfigChange = (field, value) => {
    setGenerationConfig(prev => ({ ...prev, [field]: value }));
  };

  const addBatchPrompt = () => {
    setBatchPrompts(prev => [...prev, '']);
  };

  const updateBatchPrompt = (index, value) => {
    setBatchPrompts(prev => prev.map((p, i) => i === index ? value : p));
  };

  const removeBatchPrompt = (index) => {
    setBatchPrompts(prev => prev.filter((_, i) => i !== index));
  };

  const clearConversation = () => {
    setConversationHistory([]);
  };

  const copyToClipboard = (text) => {
    navigator.clipboard.writeText(text);
  };

  const exportConversation = () => {
    const content = conversationHistory.map(msg => 
      `[${msg.timestamp.toLocaleString()}] ${msg.type.toUpperCase()}: ${msg.content}`
    ).join('\n\n');
    
    const blob = new Blob([content], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `conversation_${new Date().toISOString().split('T')[0]}.txt`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <Box sx={{ p: 3 }}>
      <Typography variant="h4" gutterBottom>
        Model Inference
      </Typography>
      
      <Typography variant="body1" color="text.secondary" sx={{ mb: 3 }}>
        Test your trained models with interactive text generation. Configure parameters and chat with your models.
      </Typography>

      <Grid container spacing={3}>
        {/* Model Selection & Configuration */}
        <Grid item xs={12} lg={4}>
          <Paper sx={{ p: 3, mb: 3 }}>
            <Typography variant="h6" gutterBottom>
              Model Selection
            </Typography>

            <FormControl fullWidth sx={{ mb: 2 }}>
              <InputLabel>Select Model</InputLabel>
              <Select
                value={selectedModel}
                onChange={(e) => setSelectedModel(e.target.value)}
                label="Select Model"
                disabled={modelsLoading}
              >
                {modelsData.map((model) => (
                  <MenuItem key={model.id} value={model.id}>
                    {model.name}
                    {model.loaded && <Chip label="Loaded" size="small" sx={{ ml: 1 }} />}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>

            {selectedModel && (
              <Box sx={{ mb: 2 }}>
                <Button
                  variant="outlined"
                  onClick={() => loadModelMutation.mutate(selectedModel)}
                  disabled={loadModelMutation.isPending}
                  sx={{ mr: 1 }}
                >
                  Load Model
                </Button>
                <Button
                  variant="outlined"
                  color="error"
                  onClick={() => unloadModelMutation.mutate(selectedModel)}
                  disabled={unloadModelMutation.isPending}
                >
                  Unload
                </Button>
              </Box>
            )}

            {(loadModelMutation.isError || unloadModelMutation.isError) && (
              <Alert severity="error" sx={{ mb: 2 }}>
                {loadModelMutation.error?.message || unloadModelMutation.error?.message}
              </Alert>
            )}
          </Paper>

          {/* Generation Configuration */}
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Generation Settings
            </Typography>

            <Accordion>
              <AccordionSummary expandIcon={<ExpandMore />}>
                <Typography>Basic Parameters</Typography>
              </AccordionSummary>
              <AccordionDetails>
                <Box sx={{ mb: 2 }}>
                  <Typography gutterBottom>Max Length: {generationConfig.max_length}</Typography>
                  <Slider
                    value={generationConfig.max_length}
                    onChange={(_, value) => handleConfigChange('max_length', value)}
                    min={10}
                    max={500}
                    step={10}
                  />
                </Box>

                <Box sx={{ mb: 2 }}>
                  <Typography gutterBottom>Temperature: {generationConfig.temperature}</Typography>
                  <Slider
                    value={generationConfig.temperature}
                    onChange={(_, value) => handleConfigChange('temperature', value)}
                    min={0.1}
                    max={2.0}
                    step={0.1}
                  />
                </Box>

                <Box sx={{ mb: 2 }}>
                  <Typography gutterBottom>Top-p: {generationConfig.top_p}</Typography>
                  <Slider
                    value={generationConfig.top_p}
                    onChange={(_, value) => handleConfigChange('top_p', value)}
                    min={0.1}
                    max={1.0}
                    step={0.1}
                  />
                </Box>
              </AccordionDetails>
            </Accordion>

            <Accordion>
              <AccordionSummary expandIcon={<ExpandMore />}>
                <Typography>Advanced Parameters</Typography>
              </AccordionSummary>
              <AccordionDetails>
                <TextField
                  fullWidth
                  label="Top-k"
                  type="number"
                  value={generationConfig.top_k}
                  onChange={(e) => handleConfigChange('top_k', parseInt(e.target.value))}
                  sx={{ mb: 2 }}
                  inputProps={{ min: 1, max: 100 }}
                />

                <Box sx={{ mb: 2 }}>
                  <Typography gutterBottom>Repetition Penalty: {generationConfig.repetition_penalty}</Typography>
                  <Slider
                    value={generationConfig.repetition_penalty}
                    onChange={(_, value) => handleConfigChange('repetition_penalty', value)}
                    min={1.0}
                    max={2.0}
                    step={0.1}
                  />
                </Box>

                <FormControlLabel
                  control={
                    <Switch
                      checked={generationConfig.do_sample}
                      onChange={(e) => handleConfigChange('do_sample', e.target.checked)}
                    />
                  }
                  label="Sampling"
                />
              </AccordionDetails>
            </Accordion>
          </Paper>
        </Grid>

        {/* Chat Interface */}
        <Grid item xs={12} lg={8}>
          <Paper sx={{ p: 3, height: '70vh', display: 'flex', flexDirection: 'column' }}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
              <Typography variant="h6">
                Chat Interface
              </Typography>
              <Box>
                <Button
                  size="small"
                  startIcon={<Download />}
                  onClick={exportConversation}
                  disabled={conversationHistory.length === 0}
                  sx={{ mr: 1 }}
                >
                  Export
                </Button>
                <Button
                  size="small"
                  startIcon={<Clear />}
                  onClick={clearConversation}
                  disabled={conversationHistory.length === 0}
                >
                  Clear
                </Button>
              </Box>
            </Box>

            {/* Conversation History */}
            <Box sx={{ flexGrow: 1, overflow: 'auto', mb: 2, border: 1, borderColor: 'divider', borderRadius: 1, p: 2 }}>
              {conversationHistory.length === 0 ? (
                <Typography color="text.secondary" sx={{ textAlign: 'center', mt: 4 }}>
                  Start a conversation by typing a message below
                </Typography>
              ) : (
                conversationHistory.map((message, index) => (
                  <Box key={index} sx={{ mb: 2 }}>
                    <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                      <Chip
                        label={message.type}
                        color={message.type === 'user' ? 'primary' : 'secondary'}
                        size="small"
                      />
                      <Typography variant="caption" sx={{ ml: 1, color: 'text.secondary' }}>
                        {message.timestamp.toLocaleTimeString()}
                      </Typography>
                      <IconButton
                        size="small"
                        onClick={() => copyToClipboard(message.content)}
                        sx={{ ml: 'auto' }}
                      >
                        <ContentCopy fontSize="small" />
                      </IconButton>
                    </Box>
                    <Paper sx={{ p: 2, bgcolor: message.type === 'user' ? 'primary.50' : 'grey.50' }}>
                      <Typography variant="body2" sx={{ whiteSpace: 'pre-wrap' }}>
                        {message.content}
                      </Typography>
                      {message.metadata && (
                        <Box sx={{ mt: 1, display: 'flex', gap: 2 }}>
                          <Typography variant="caption" color="text.secondary">
                            Time: {message.metadata.generation_time?.toFixed(2)}s
                          </Typography>
                          <Typography variant="caption" color="text.secondary">
                            Tokens: {message.metadata.tokens_generated}
                          </Typography>
                        </Box>
                      )}
                    </Paper>
                  </Box>
                ))
              )}
            </Box>

            {/* Input Area */}
            <Box sx={{ display: 'flex', gap: 1 }}>
              <TextField
                fullWidth
                multiline
                rows={2}
                placeholder="Type your message here..."
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                onKeyPress={(e) => {
                  if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    handleGenerate();
                  }
                }}
                disabled={!selectedModel || generateMutation.isPending}
              />
              <Button
                variant="contained"
                onClick={handleGenerate}
                disabled={!prompt.trim() || !selectedModel || generateMutation.isPending}
                startIcon={generateMutation.isPending ? <Stop /> : <Send />}
                sx={{ minWidth: 100 }}
              >
                {generateMutation.isPending ? 'Stop' : 'Send'}
              </Button>
            </Box>

            {generateMutation.isError && (
              <Alert severity="error" sx={{ mt: 2 }}>
                Generation failed: {generateMutation.error?.message}
              </Alert>
            )}
          </Paper>
        </Grid>
      </Grid>

      {/* Batch Generation */}
      <Paper sx={{ p: 3, mt: 3 }}>
        <Typography variant="h6" gutterBottom>
          Batch Generation
        </Typography>
        
        <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
          Generate responses for multiple prompts at once
        </Typography>

        <List>
          {batchPrompts.map((prompt, index) => (
            <ListItem key={index} sx={{ px: 0 }}>
              <TextField
                fullWidth
                placeholder={`Prompt ${index + 1}`}
                value={prompt}
                onChange={(e) => updateBatchPrompt(index, e.target.value)}
                multiline
                rows={2}
              />
              <ListItemSecondaryAction>
                <IconButton
                  edge="end"
                  onClick={() => removeBatchPrompt(index)}
                  disabled={batchPrompts.length === 1}
                >
                  <Clear />
                </IconButton>
              </ListItemSecondaryAction>
            </ListItem>
          ))}
        </List>

        <Box sx={{ mt: 2, display: 'flex', gap: 1 }}>
          <Button onClick={addBatchPrompt}>
            Add Prompt
          </Button>
          <Button
            variant="contained"
            onClick={handleBatchGenerate}
            disabled={!selectedModel || batchGenerateMutation.isPending}
            startIcon={<PlayArrow />}
          >
            Generate All
          </Button>
        </Box>

        {batchGenerateMutation.isSuccess && (
          <Box sx={{ mt: 2 }}>
            <Typography variant="subtitle2" gutterBottom>
              Batch Results:
            </Typography>
            {batchGenerateMutation.data.results.map((result, index) => (
              <Card key={index} sx={{ mb: 1 }}>
                <CardContent>
                  <Typography variant="body2" color="text.secondary" gutterBottom>
                    Prompt {index + 1}:
                  </Typography>
                  <Typography variant="body2" sx={{ mb: 1 }}>
                    {result.prompt}
                  </Typography>
                  <Divider sx={{ my: 1 }} />
                  <Typography variant="body2" color="text.secondary" gutterBottom>
                    Generated:
                  </Typography>
                  <Typography variant="body2">
                    {result.generated_text}
                  </Typography>
                </CardContent>
              </Card>
            ))}
          </Box>
        )}

        {batchGenerateMutation.isError && (
          <Alert severity="error" sx={{ mt: 2 }}>
            Batch generation failed: {batchGenerateMutation.error?.message}
          </Alert>
        )}
      </Paper>
    </Box>
  );
};

export default Inference; 