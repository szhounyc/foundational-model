import React, { useState, useCallback, useEffect } from 'react';
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
  Accordion,
  AccordionSummary,
  AccordionDetails,
  TextField,
  Tooltip,
  Badge,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
} from '@mui/material';
import {
  CloudUpload as UploadIcon,
  Delete as DeleteIcon,
  Gavel as ReviewIcon,
  Description as PdfIcon,
  CheckCircle as CheckIcon,
  Error as ErrorIcon,
  Business as BusinessIcon,
  Comment as CommentIcon,
  AutoAwesome as AIIcon,
  ExpandMore as ExpandMoreIcon,
  Warning as WarningIcon,
  Info as InfoIcon,
  Assignment as TemplateIcon,
  Compare as CompareIcon,
  Close as CloseIcon,
  Category as CategoryIcon,
  Save as SaveIcon,
  Download as DownloadIcon,
} from '@mui/icons-material';
import { useDropzone } from 'react-dropzone';
import { useMutation, useQuery } from 'react-query';
import axios from 'axios';

const API_BASE = process.env.REACT_APP_API_URL || 'http://localhost:9100';

const COMMENT_SEVERITY_COLORS = {
  'high': 'error',
  'medium': 'warning', 
  'low': 'info',
  'suggestion': 'success'
};

const COMMENT_SEVERITY_ICONS = {
  'high': ErrorIcon,
  'medium': WarningIcon,
  'low': InfoIcon,
  'suggestion': CheckIcon
};

const CONTRACT_TYPES = [
  'Purchase Agreement',
  'Rider',
  'Legal Comments'
];

const DOCUMENT_TYPES = [
  { label: 'Purchase Agreement', value: 'purchase_agreement' },
  { label: 'Rider', value: 'rider' },
  { label: 'Legal Comments', value: 'legal_comments' }
];

// Diff View Component
const DiffViewer = ({ differences, templateFilename, contractFilename, onClose }) => {
  // Group differences by line number for better display
  const groupedDiffs = differences.reduce((acc, diff) => {
    const lineNum = diff.line_number;
    if (!acc[lineNum]) {
      acc[lineNum] = { removed: [], added: [] };
    }
    if (diff.type === 'removed') {
      acc[lineNum].removed.push(diff);
    } else if (diff.type === 'added') {
      acc[lineNum].added.push(diff);
    }
    return acc;
  }, {});

  const sortedLineNumbers = Object.keys(groupedDiffs).sort((a, b) => parseInt(a) - parseInt(b));

  return (
    <Dialog open={true} onClose={onClose} maxWidth="xl" fullWidth>
      <DialogTitle>
        <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <CompareIcon />
            <Typography variant="h6">Document Comparison</Typography>
          </Box>
          <IconButton onClick={onClose}>
            <CloseIcon />
          </IconButton>
        </Box>
      </DialogTitle>
      <DialogContent>
        <Box sx={{ mb: 2 }}>
          <Grid container spacing={2}>
            <Grid item xs={6}>
              <Paper sx={{ p: 2, bgcolor: 'error.light', color: 'error.dark' }}>
                <Typography variant="subtitle1" sx={{ fontWeight: 600 }}>
                  Template: {templateFilename}
                </Typography>
                <Typography variant="body2">
                  Lines removed from template
                </Typography>
              </Paper>
            </Grid>
            <Grid item xs={6}>
              <Paper sx={{ p: 2, bgcolor: 'success.light', color: 'success.dark' }}>
                <Typography variant="subtitle1" sx={{ fontWeight: 600 }}>
                  Contract: {contractFilename}
                </Typography>
                <Typography variant="body2">
                  Lines added in contract
                </Typography>
              </Paper>
            </Grid>
          </Grid>
        </Box>

        <Paper sx={{ maxHeight: '70vh', overflow: 'auto' }}>
          {sortedLineNumbers.slice(0, 100).map((lineNum) => {
            const diffs = groupedDiffs[lineNum];
            return (
              <Box key={lineNum} sx={{ borderBottom: '1px solid #e0e0e0' }}>
                <Grid container>
                  {/* Template side (removed lines) */}
                  <Grid item xs={6} sx={{ borderRight: '1px solid #e0e0e0' }}>
                    {diffs.removed.map((diff, idx) => (
                      <Box
                        key={idx}
                        sx={{
                          p: 1,
                          bgcolor: 'error.light',
                          fontFamily: 'monospace',
                          fontSize: '0.875rem',
                          whiteSpace: 'pre-wrap',
                          wordBreak: 'break-word',
                        }}
                      >
                        <Typography variant="caption" sx={{ color: 'error.dark', mr: 1 }}>
                          -{lineNum}:
                        </Typography>
                        <span style={{ color: '#d32f2f' }}>{diff.content}</span>
                      </Box>
                    ))}
                  </Grid>

                  {/* Contract side (added lines) */}
                  <Grid item xs={6}>
                    {diffs.added.map((diff, idx) => (
                      <Box
                        key={idx}
                        sx={{
                          p: 1,
                          bgcolor: 'success.light',
                          fontFamily: 'monospace',
                          fontSize: '0.875rem',
                          whiteSpace: 'pre-wrap',
                          wordBreak: 'break-word',
                        }}
                      >
                        <Typography variant="caption" sx={{ color: 'success.dark', mr: 1 }}>
                          +{lineNum}:
                        </Typography>
                        <span style={{ color: '#2e7d32' }}>{diff.content}</span>
                      </Box>
                    ))}
                  </Grid>
                </Grid>
              </Box>
            );
          })}
          {sortedLineNumbers.length > 100 && (
            <Box sx={{ p: 2, textAlign: 'center', bgcolor: 'grey.100' }}>
              <Typography variant="body2" color="text.secondary">
                Showing first 100 differences. Total: {differences.length} differences.
              </Typography>
            </Box>
          )}
        </Paper>
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose}>Close</Button>
      </DialogActions>
    </Dialog>
  );
};

function ContractReview() {
  const [uploadedFiles, setUploadedFiles] = useState([]);
  const [selectedFiles, setSelectedFiles] = useState([]);
  const [reviewType, setReviewType] = useState('comprehensive');
  const [modelId, setModelId] = useState('sftj-s1xkr35z');
  const [reviewResult, setReviewResult] = useState(null);
  const [selectedLawFirm, setSelectedLawFirm] = useState('');
  const [contractType, setContractType] = useState('');
  const [analysisNotes, setAnalysisNotes] = useState('');
  const [commentsResults, setCommentsResults] = useState([]);
  const [detectedLawFirms, setDetectedLawFirms] = useState([]);
  const [expandedPanel, setExpandedPanel] = useState(false);
  
  // New state for view toggles
  const [diffViewOpen, setDiffViewOpen] = useState(false);
  const [selectedComparison, setSelectedComparison] = useState(null);
  const [fileDocumentTypes, setFileDocumentTypes] = useState({}); // New state for document types
  const [commentGenerationProgress, setCommentGenerationProgress] = useState({ current: 0, total: 0 }); // New state for progress

  // Fetch contract files
  const { data: contractFiles = [], refetch: refetchFiles } = useQuery(
    'contract-files',
    async () => {
      const response = await axios.get(`${API_BASE}/api/contracts/files`);
      return response.data.files || [];
    },
    {
      refetchInterval: 30000,
    }
  );

  // Fetch law firms
  const { data: lawFirmsData } = useQuery(
    'law-firms',
    async () => {
      const response = await axios.get(`${API_BASE}/api/law-firms`);
      return response.data;
    }
  );

  // Fetch models
  const { data: modelsData } = useQuery(
    'models',
    async () => {
      const response = await axios.get(`${API_BASE}/api/inference/models`);
      return response.data;
    }
  );

  // Update uploaded files when contract files change
  useEffect(() => {
    setUploadedFiles(contractFiles);
  }, [contractFiles]);

  // New mutation for updating document types
  const updateDocumentTypeMutation = useMutation(
    async ({ fileId, documentType }) => {
      const response = await axios.put(`${API_BASE}/api/contracts/files/${fileId}/document-type`, {
        document_type: documentType
      });
      return response.data;
    },
    {
      onSuccess: (data, variables) => {
        // Update local state
        setFileDocumentTypes(prev => ({
          ...prev,
          [variables.fileId]: variables.documentType
        }));
        
        // Update uploaded files list
        setUploadedFiles(prev => prev.map(file => 
          file.id === variables.fileId 
            ? { ...file, document_type: variables.documentType }
            : file
        ));
      },
      onError: (error) => {
        console.error('Failed to update document type:', error);
      }
    }
  );

  // Function to handle document type change
  const handleDocumentTypeChange = (fileId, documentType) => {
    updateDocumentTypeMutation.mutate({ fileId, documentType });
  };

  // Upload mutation
  const uploadMutation = useMutation(
    async (file) => {
      const formData = new FormData();
      formData.append('file', file);
      
      const response = await axios.post(`${API_BASE}/api/contracts/upload`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      
      return response.data;
    },
    {
      onSuccess: (data) => {
        setUploadedFiles(prev => [...prev, data]);
        // Initialize document type state for new file
        setFileDocumentTypes(prev => ({
          ...prev,
          [data.id]: data.document_type || ''
        }));
      },
      onError: (error) => {
        console.error('Upload failed:', error);
      }
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

  // Comment generation mutation - updated to handle multiple files
  const commentsMutation = useMutation(
    async (data) => {
      const response = await axios.post(`${API_BASE}/api/contracts/comment/batch`, data);
      return response.data;
    },
    {
      onSuccess: (data) => {
        setCommentsResults([data]); // Single result with all documents
        setCommentGenerationProgress({ current: 1, total: 1 });
      },
      onError: (error) => {
        console.error('Comment generation failed:', error);
      }
    }
  );

  // Auto-detect law firms from contract text
  const detectLawFirms = (text) => {
    const lawFirms = lawFirmsData?.law_firms || [];
    const detected = [];
    
    lawFirms.forEach(firm => {
      const hasMatch = firm.keywords.some(keyword => 
        text.toLowerCase().includes(keyword.toLowerCase())
      );
      if (hasMatch) {
        detected.push(firm);
      }
    });
    
    setDetectedLawFirms(detected);
    
    // Auto-select the first detected law firm
    if (detected.length > 0 && !selectedLawFirm) {
      setSelectedLawFirm(detected[0].id);
    }
  };

  const onDrop = useCallback((acceptedFiles) => {
    acceptedFiles.forEach((file) => {
      if (file.type === 'application/pdf' || file.type === 'application/vnd.openxmlformats-officedocument.wordprocessingml.document') {
        uploadMutation.mutate(file);
      }
    });
  }, [uploadMutation]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'application/pdf': ['.pdf'],
      'application/vnd.openxmlformats-officedocument.wordprocessingml.document': ['.docx'],
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

  const handleGenerateComments = async () => {
    if (selectedFiles.length === 0) {
      return;
    }

    // Reset previous results and set up progress tracking
    setCommentsResults([]);
    setCommentGenerationProgress({ current: 0, total: 1 });

    try {
      await commentsMutation.mutateAsync({
        contract_file_ids: selectedFiles,
        law_firm_id: selectedLawFirm || undefined
      });
    } catch (error) {
      console.error('Failed to generate comments:', error);
    }
  };

  const handlePanelChange = (panel) => (event, isExpanded) => {
    setExpandedPanel(isExpanded ? panel : false);
  };

  const formatFileSize = (bytes) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const getLawFirmName = (lawFirmId) => {
    const firm = lawFirmsData?.law_firms?.find(f => f.id === lawFirmId);
    return firm ? firm.name : 'Unknown';
  };

  const lawFirms = lawFirmsData?.law_firms || [];

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
      console.error('Failed to download template:', error);
    }
  };

  return (
    <Box sx={{ maxWidth: 1200, mx: 'auto' }}>
      <Typography variant="h4" gutterBottom sx={{ mb: 3, fontWeight: 600 }}>
        Contract Review & AI Comments
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
                  {isDragActive ? 'Drop PDF or DOCX files here' : 'Drag & drop PDF or DOCX files here'}
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
                Analysis Configuration
              </Typography>

              {/* Law Firm Detection/Selection */}
              <Box sx={{ mb: 2 }}>
                {detectedLawFirms.length > 0 && (
                  <Alert severity="info" sx={{ mb: 2 }}>
                    <Typography variant="subtitle2" gutterBottom>
                      Auto-detected Law Firms:
                    </Typography>
                    <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
                      {detectedLawFirms.map(firm => (
                        <Chip
                          key={firm.id}
                          label={firm.name}
                          icon={<BusinessIcon />}
                          color="primary"
                          variant="outlined"
                          size="small"
                        />
                      ))}
                    </Box>
                  </Alert>
                )}

                <FormControl fullWidth sx={{ mb: 2 }}>
                  <InputLabel>Law Firm</InputLabel>
                  <Select
                    value={selectedLawFirm}
                    label="Law Firm"
                    onChange={(e) => setSelectedLawFirm(e.target.value)}
                  >
                    <MenuItem value="">
                      <em>Select Law Firm</em>
                    </MenuItem>
                    {lawFirms.map((firm) => (
                      <MenuItem key={firm.id} value={firm.id}>
                        <Box sx={{ display: 'flex', alignItems: 'center' }}>
                          <BusinessIcon sx={{ mr: 1, fontSize: 16 }} />
                          {firm.name}
                          {detectedLawFirms.some(d => d.id === firm.id) && (
                            <Chip label="Auto-detected" size="small" color="success" sx={{ ml: 1 }} />
                          )}
                        </Box>
                      </MenuItem>
                    ))}
                  </Select>
                </FormControl>
              </Box>

              <FormControl fullWidth sx={{ mb: 2 }}>
                <InputLabel>Contract Type</InputLabel>
                <Select
                  value={contractType}
                  label="Contract Type"
                  onChange={(e) => setContractType(e.target.value)}
                >
                  {CONTRACT_TYPES.map((type) => (
                    <MenuItem key={type} value={type}>
                      {type}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>

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

              <TextField
                fullWidth
                label="Analysis Notes (Optional)"
                value={analysisNotes}
                onChange={(e) => setAnalysisNotes(e.target.value)}
                multiline
                rows={2}
                placeholder="Any specific areas you'd like the AI to focus on..."
                sx={{ mb: 3 }}
              />

              {/* Action Buttons */}
              <Grid container spacing={2}>
                <Grid item xs={12} sm={6}>
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
                </Grid>
                <Grid item xs={12} sm={6}>
                  <Button
                    variant="contained"
                    size="large"
                    fullWidth
                    startIcon={<AIIcon />}
                    onClick={handleGenerateComments}
                    disabled={selectedFiles.length === 0 || commentsMutation.isLoading || !selectedLawFirm}
                    sx={{ 
                      py: 1.5,
                      background: 'linear-gradient(45deg, #6366f1, #8b5cf6)',
                      '&:hover': {
                        background: 'linear-gradient(45deg, #5855eb, #7c3aed)',
                      }
                    }}
                  >
                    {commentsMutation.isLoading ? 'Generating...' : 'Generate Comments'}
                  </Button>
                </Grid>
              </Grid>

              {(reviewMutation.isLoading || commentsMutation.isLoading) && (
                <Box sx={{ mt: 2 }}>
                  <LinearProgress />
                  <Typography variant="body2" sx={{ mt: 1 }}>
                    {reviewMutation.isLoading ? 'AI is analyzing your contracts...' : 'AI is generating comments...'}
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
                        flexDirection: 'column',
                        alignItems: 'stretch',
                        p: 2,
                      }}
                    >
                      <Box 
                        sx={{ display: 'flex', alignItems: 'center', width: '100%', mb: 1 }}
                        onClick={() => handleFileSelect(file.id)}
                      >
                        <PdfIcon color="error" sx={{ mr: 2 }} />
                        <Box sx={{ flexGrow: 1 }}>
                          <Typography variant="subtitle1" sx={{ fontWeight: 500 }}>
                            {file.filename}
                          </Typography>
                          <Box sx={{ display: 'flex', gap: 1, alignItems: 'center', mt: 0.5 }}>
                            <Chip
                              size="small"
                              icon={file.text_extracted ? <CheckIcon /> : <ErrorIcon />}
                              label={file.text_extracted ? 'Text Extracted' : 'Processing Failed'}
                              color={file.text_extracted ? 'success' : 'error'}
                            />
                            <Typography variant="caption">
                              {formatFileSize(file.size_bytes || 0)}
                            </Typography>
                            {file.text_length && (
                              <Typography variant="caption">
                                • {file.text_length} characters
                              </Typography>
                            )}
                            <Typography variant="caption" color="text.secondary">
                              • Uploaded: {new Date(file.uploaded_at).toLocaleDateString()}
                            </Typography>
                          </Box>
                        </Box>
                        <IconButton
                          edge="end"
                          onClick={(e) => {
                            e.stopPropagation();
                            handleRemoveFile(file.id);
                          }}
                        >
                          <DeleteIcon />
                        </IconButton>
                      </Box>
                      
                      {/* Document Type Selection */}
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mt: 1 }}>
                        <CategoryIcon sx={{ color: 'text.secondary' }} />
                        <FormControl size="small" sx={{ minWidth: 200 }}>
                          <InputLabel>Document Type</InputLabel>
                          <Select
                            value={file.document_type || ''}
                            label="Document Type"
                            onChange={(e) => {
                              e.stopPropagation();
                              handleDocumentTypeChange(file.id, e.target.value);
                            }}
                            onClick={(e) => e.stopPropagation()}
                          >
                            <MenuItem value="">
                              <em>Select Type</em>
                            </MenuItem>
                            {DOCUMENT_TYPES.map((type) => (
                              <MenuItem key={type.value} value={type.value}>
                                {type.label}
                              </MenuItem>
                            ))}
                          </Select>
                        </FormControl>
                        {file.document_type && (
                          <Chip
                            label={DOCUMENT_TYPES.find(t => t.value === file.document_type)?.label || file.document_type}
                            size="small"
                            color="primary"
                            variant="outlined"
                            icon={<CategoryIcon />}
                          />
                        )}
                        {updateDocumentTypeMutation.isLoading && (
                          <Chip
                            label="Saving..."
                            size="small"
                            color="info"
                            icon={<SaveIcon />}
                          />
                        )}
                      </Box>
                    </ListItem>
                  ))}
                </List>
                {selectedFiles.length > 0 && (
                  <Alert severity="info" sx={{ mt: 2 }}>
                    {selectedFiles.length} file(s) selected for analysis
                  </Alert>
                )}
              </CardContent>
            </Card>
          </Grid>
        )}

        {/* Results Section */}
        {(reviewResult || commentsResults.length > 0) && (
          <Grid item xs={12}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Analysis Results
                </Typography>

                {/* Review Results Accordion */}
                {reviewResult && (
                  <Accordion 
                    expanded={expandedPanel === 'review'} 
                    onChange={handlePanelChange('review')}
                    sx={{ mb: 2 }}
                  >
                    <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                      <Box sx={{ display: 'flex', alignItems: 'center', width: '100%' }}>
                        <ReviewIcon sx={{ mr: 2, color: 'primary.main' }} />
                        <Box sx={{ flexGrow: 1 }}>
                          <Typography variant="h6">Contract Review</Typography>
                          <Box sx={{ display: 'flex', gap: 1, mt: 1 }}>
                            <Chip
                              label={`Status: ${reviewResult.status}`}
                              color="success"
                              size="small"
                            />
                            <Chip
                              label={`Processing Time: ${reviewResult.processing_time?.toFixed(1)}s`}
                              variant="outlined"
                              size="small"
                            />
                          </Box>
                        </Box>
                      </Box>
                    </AccordionSummary>
                    <AccordionDetails>
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
                    </AccordionDetails>
                  </Accordion>
                )}

                {/* Comments Results Accordion */}
                {commentsResults.length > 0 && (
                  <Accordion 
                    expanded={expandedPanel === 'comments'} 
                    onChange={handlePanelChange('comments')}
                    sx={{ mb: 2 }}
                  >
                    <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                      <Box sx={{ display: 'flex', alignItems: 'center', width: '100%' }}>
                        <AIIcon sx={{ mr: 2, color: 'purple' }} />
                        <Box sx={{ flexGrow: 1 }}>
                          <Typography variant="h6">AI Comments & Analysis</Typography>
                          <Box sx={{ display: 'flex', gap: 1, mt: 1 }}>
                            <Chip
                              label={getLawFirmName(selectedLawFirm)}
                              icon={<BusinessIcon />}
                              size="small"
                              variant="outlined"
                            />
                            <Chip
                              label={`${commentsResults.length > 0 && commentsResults[0].contract_files ? commentsResults[0].contract_files.length : 0} Documents Analyzed`}
                              size="small"
                              color="primary"
                              variant="outlined"
                            />
                            <Chip
                              label={`${commentsResults.length > 0 && commentsResults[0].comments ? commentsResults[0].comments.length : 0} Legal Comments`}
                              icon={<CommentIcon />}
                              size="small"
                              variant="outlined"
                            />
                            <Chip
                              label={`${commentsResults.reduce((total, result) => total + (result.document_comparisons?.reduce((docTotal, doc) => docTotal + (doc.comparisons?.length || 0), 0) || 0), 0)} Template Comparisons`}
                              icon={<CompareIcon />}
                              size="small"
                              variant="outlined"
                            />
                          </Box>
                        </Box>
                      </Box>
                    </AccordionSummary>
                    <AccordionDetails>
                      {commentsResults.length > 0 ? (
                        <Box>
                          {/* Progress indicator */}
                          {commentsMutation.isLoading && (
                            <Box sx={{ mb: 3 }}>
                              <LinearProgress
                                variant="determinate"
                                value={(commentGenerationProgress.current / commentGenerationProgress.total) * 100}
                              />
                              <Typography variant="body2" sx={{ mt: 1, color: 'text.secondary' }}>
                                Processing {commentGenerationProgress.current} of {commentGenerationProgress.total} documents...
                              </Typography>
                            </Box>
                          )}

                          {/* Display results for each document */}
                          {commentsResults.map((result, resultIndex) => (
                            <Box key={result.comment_id} sx={{ mb: 4 }}>
                              {/* Overall Comments Section (shown once) */}
                              {result.comments && result.comments.length > 0 && (
                                <Box sx={{ mb: 4 }}>
                                  <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                                    <Typography variant="h6" sx={{ display: 'flex', alignItems: 'center' }}>
                                      <CommentIcon sx={{ mr: 1 }} />
                                      Legal Comments
                                    </Typography>
                                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                                      {result.legal_comments_template_id && (
                                        <Tooltip title={`Download template: ${result.legal_comments_template_filename || 'Legal Comments Template'}`}>
                                          <IconButton
                                            size="small"
                                            color="primary"
                                            onClick={() => handleDownloadTemplate(result.legal_comments_template_id, result.legal_comments_template_filename)}
                                          >
                                            <DownloadIcon />
                                          </IconButton>
                                        </Tooltip>
                                      )}
                                    </Box>
                                  </Box>

                                  {/* Compact View */}
                                  {result.comments.map((comment, index) => (
                                    <Paper sx={{ 
                                      p: 4, 
                                      bgcolor: 'white', 
                                      maxHeight: '60vh', 
                                      overflow: 'auto',
                                      border: '1px solid #e0e0e0',
                                      boxShadow: '0 2px 8px rgba(0,0,0,0.1)'
                                    }} key={index}>
                                      <Typography 
                                        variant="body1" 
                                        sx={{ 
                                          whiteSpace: 'pre-wrap', 
                                          lineHeight: 1.6, 
                                          fontFamily: '"Times New Roman", Times, serif',
                                          fontSize: '14px',
                                          color: '#333',
                                          textAlign: 'justify'
                                        }}
                                      >
                                        {comment.comment}
                                      </Typography>
                                    </Paper>
                                  ))}
                                </Box>
                              )}

                              {/* Document Comparisons Section */}
                              {result.document_comparisons && result.document_comparisons.length > 0 && (
                                <Box sx={{ mt: 4 }}>
                                  <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center' }}>
                                    <CompareIcon sx={{ mr: 1 }} />
                                    Document Comparisons
                                  </Typography>
                                  
                                  {result.document_comparisons.map((docComparison, docIndex) => (
                                    <Box key={docComparison.contract_file_id} sx={{ mb: 3 }}>
                                      <Typography variant="subtitle1" sx={{ fontWeight: 600, mb: 2, display: 'flex', alignItems: 'center' }}>
                                        <PdfIcon sx={{ mr: 1 }} />
                                        {docComparison.contract_filename}
                                        <Chip
                                          label={docComparison.document_type || 'Unknown Type'}
                                          size="small"
                                          variant="outlined"
                                          sx={{ ml: 2 }}
                                        />
                                      </Typography>
                                      
                                      {docComparison.comparisons && docComparison.comparisons.length > 0 ? (
                                        <Grid container spacing={2}>
                                          {docComparison.comparisons.map((comparison, compIndex) => (
                                            <Grid item xs={12} key={compIndex}>
                                              <Card variant="outlined" sx={{ bgcolor: 'info.light' }}>
                                                <CardContent>
                                                  <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', mb: 2 }}>
                                                    <Box sx={{ flexGrow: 1 }}>
                                                      <Typography variant="subtitle2" sx={{ fontWeight: 600, color: 'info.dark' }}>
                                                        {comparison.template_type?.replace('_', ' ').toUpperCase() || 'Template'} Comparison
                                                      </Typography>
                                                      <Typography variant="body2" sx={{ color: 'info.dark', mt: 1 }}>
                                                        Template: {comparison.template_filename}
                                                      </Typography>
                                                      <Box sx={{ mt: 2 }}>
                                                        <Chip
                                                          label={`${comparison.differences?.length || 0} differences found`}
                                                          size="small"
                                                          color={comparison.differences?.length > 0 ? 'warning' : 'success'}
                                                          sx={{ mr: 1 }}
                                                        />
                                                        <Chip
                                                          label={comparison.template_type || 'Unknown Type'}
                                                          size="small"
                                                          variant="outlined"
                                                        />
                                                      </Box>
                                                    </Box>
                                                    
                                                    {comparison.differences && comparison.differences.length > 0 && (
                                                      <Button
                                                        variant="contained"
                                                        size="small"
                                                        startIcon={<CompareIcon />}
                                                        onClick={() => {
                                                          setSelectedComparison({
                                                            ...comparison,
                                                            contract_filename: docComparison.contract_filename
                                                          });
                                                          setDiffViewOpen(true);
                                                        }}
                                                        sx={{ ml: 2 }}
                                                      >
                                                        View Diff
                                                      </Button>
                                                    )}
                                                  </Box>
                                                  
                                                  {comparison.summary && (
                                                    <Typography variant="body2" sx={{ color: 'info.dark', fontStyle: 'italic' }}>
                                                      {comparison.summary}
                                                    </Typography>
                                                  )}
                                                </CardContent>
                                              </Card>
                                            </Grid>
                                          ))}
                                        </Grid>
                                      ) : (
                                        <Paper sx={{ p: 2, bgcolor: 'grey.100' }}>
                                          <Typography variant="body2" color="text.secondary">
                                            No template comparisons available for this document type.
                                          </Typography>
                                        </Paper>
                                      )}
                                    </Box>
                                  ))}
                                </Box>
                              )}
                            </Box>
                          ))}
                        </Box>
                      ) : (
                        <Paper sx={{ p: 3, textAlign: 'center', bgcolor: 'grey.50' }}>
                          <CommentIcon sx={{ fontSize: 48, color: 'text.secondary', mb: 2 }} />
                          <Typography variant="body1" color="text.secondary">
                            No AI comments available for these contracts yet.
                          </Typography>
                        </Paper>
                      )}
                    </AccordionDetails>
                  </Accordion>
                )}
              </CardContent>
            </Card>
          </Grid>
        )}

        {/* Error Messages */}
        {reviewMutation.isError && (
          <Grid item xs={12}>
            <Alert severity="error">
              Review failed: {reviewMutation.error?.response?.data?.detail || 'Unknown error'}
            </Alert>
          </Grid>
        )}

        {commentsMutation.isError && (
          <Grid item xs={12}>
            <Alert severity="error">
              Comment generation failed: {commentsMutation.error?.response?.data?.detail || 'Unknown error'}
            </Alert>
          </Grid>
        )}
      </Grid>

      {/* Diff Viewer Dialog */}
      {diffViewOpen && selectedComparison && (
        <DiffViewer
          differences={selectedComparison.differences}
          templateFilename={selectedComparison.template_filename}
          contractFilename={selectedComparison.contract_filename || 'Contract'}
          onClose={() => {
            setDiffViewOpen(false);
            setSelectedComparison(null);
          }}
        />
      )}
    </Box>
  );
}

export default ContractReview; 