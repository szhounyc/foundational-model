import React, { useState } from 'react';
import { useQuery } from 'react-query';
import {
  Box,
  Typography,
  Paper,
  Tabs,
  Tab,
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Button,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Chip,
  Badge,
  Grid,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Divider,
  Alert,
  CircularProgress,
  Card,
  CardContent,
  ToggleButton,
  ToggleButtonGroup,
  IconButton
} from '@mui/material';
import {
  Search as SearchIcon,
  FilterList as FilterIcon,
  Refresh as RefreshIcon,
  ExpandMore as ExpandMoreIcon,
  Close as CloseIcon,
  RateReview as ReviewIcon,
  SmartToy as AIIcon,
  Business as BusinessIcon,
  Comment as CommentIcon,
  Schedule as ScheduleIcon,
  CheckCircle as CheckIcon,
  Error as ErrorIcon,
  HourglassEmpty as ProcessingIcon,
  Description as FileIcon,
  Compare as CompareIcon,
  Visibility as ViewIcon,
  ViewList as ListViewIcon,
  ViewModule as CompactViewIcon,
  Warning as WarningIcon,
  Info as InfoIcon
} from '@mui/icons-material';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:9100';

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

// Diff View Component - same as ContractReview
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
                Showing first 100 differences. Total: {sortedLineNumbers.length}
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

const ReviewHistory = () => {
  const [activeTab, setActiveTab] = useState(0);
  const [searchTerm, setSearchTerm] = useState('');
  const [statusFilter, setStatusFilter] = useState('all');
  const [selectedDetail, setSelectedDetail] = useState(null);
  const [detailDialogOpen, setDetailDialogOpen] = useState(false);
  
  // New state for view toggles
  const [commentsViewMode, setCommentsViewMode] = useState('list'); // 'list' or 'compact'
  const [diffViewOpen, setDiffViewOpen] = useState(false);
  const [selectedComparison, setSelectedComparison] = useState(null);

  // Fetch contract reviews
  const { data: reviews = [], isLoading: reviewsLoading, error: reviewsError, refetch: refetchReviews } = useQuery(
    'contractReviews',
    async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/api/contracts/reviews`);
        if (!response.ok) {
          throw new Error('Failed to fetch reviews');
        }
        const data = await response.json();
        // Handle the nested structure where reviews are under "reviews" key
        return data.reviews && Array.isArray(data.reviews) ? data.reviews : [];
      } catch (error) {
        console.error('Error fetching reviews:', error);
        return [];
      }
    },
    {
      refetchInterval: 30000,
      retry: 3,
      retryDelay: 1000
    }
  );

  // Fetch contract comments
  const { data: comments = [], isLoading: commentsLoading, error: commentsError, refetch: refetchComments } = useQuery(
    'contractComments',
    async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/api/contracts/comments`);
        if (!response.ok) {
          throw new Error('Failed to fetch comments');
        }
        const data = await response.json();
        // Handle the nested structure where comments are under "comments" key
        return data.comments && Array.isArray(data.comments) ? data.comments : (Array.isArray(data) ? data : []);
      } catch (error) {
        console.error('Error fetching comments:', error);
        return [];
      }
    },
    {
      refetchInterval: 30000,
      retry: 3,
      retryDelay: 1000
    }
  );

  const filterItems = (items, type) => {
    if (!Array.isArray(items)) return [];
    
    return items.filter(item => {
      if (!item) return false;
      
      const matchesSearch = !searchTerm || 
        (item.contract_filename && item.contract_filename.toLowerCase().includes(searchTerm.toLowerCase())) ||
        (item.contract_files && Array.isArray(item.contract_files) && 
         item.contract_files.some(file => file.toLowerCase().includes(searchTerm.toLowerCase()))) ||
        (item.law_firm_name && item.law_firm_name.toLowerCase().includes(searchTerm.toLowerCase())) ||
        (item.model_id && item.model_id.toLowerCase().includes(searchTerm.toLowerCase()));
      
      const matchesStatus = statusFilter === 'all' || 
        (item.status && item.status.toLowerCase() === statusFilter.toLowerCase());
      
      return matchesSearch && matchesStatus;
    });
  };

  const getStatusIcon = (status) => {
    if (!status) return <ProcessingIcon />;
    
    const statusLower = status.toLowerCase();
    switch (statusLower) {
      case 'completed':
        return <CheckIcon />;
      case 'failed':
      case 'error':
        return <ErrorIcon />;
      case 'processing':
      case 'pending':
      default:
        return <ProcessingIcon />;
    }
  };

  const getStatusColor = (status) => {
    if (!status) return 'default';
    
    const statusLower = status.toLowerCase();
    switch (statusLower) {
      case 'completed':
        return 'success';
      case 'failed':
      case 'error':
        return 'error';
      case 'processing':
      case 'pending':
      default:
        return 'warning';
    }
  };

  const formatDate = (dateString) => {
    return dateString ? new Date(dateString).toLocaleString() : 'N/A';
  };

  const formatProcessingTime = (seconds) => {
    return seconds ? `${parseFloat(seconds).toFixed(2)}s` : 'N/A';
  };

  const handleViewDetails = async (type, id) => {
    try {
      const items = type === 'review' ? reviews : comments;
      const item = items.find(i => i.id === id);
      if (item) {
        setSelectedDetail({ type, data: item });
        setDetailDialogOpen(true);
      }
    } catch (error) {
      console.error('Error viewing details:', error);
    }
  };

  const handleCloseDialog = () => {
    setDetailDialogOpen(false);
    setSelectedDetail(null);
  };

  const renderReviewsList = () => {
    const filteredReviews = filterItems(reviews, 'reviews');
    
    if (filteredReviews.length === 0) {
      return (
        <Paper sx={{ p: 4, textAlign: 'center' }}>
          <ReviewIcon sx={{ fontSize: 48, color: 'text.secondary', mb: 2 }} />
          <Typography variant="h6" color="text.secondary">
            No contract reviews found
          </Typography>
          <Typography variant="body2" color="text.secondary">
            {searchTerm || statusFilter !== 'all' ? 'Try adjusting your filters' : 'Start by uploading and reviewing some contracts'}
          </Typography>
        </Paper>
      );
    }

    return (
      <Box>
        {filteredReviews.map((review, index) => (
          <Accordion 
            key={review.id || index}
            sx={{ mb: 2 }}
          >
            <AccordionSummary expandIcon={<ExpandMoreIcon />}>
              <Box sx={{ display: 'flex', alignItems: 'center', width: '100%' }}>
                <ReviewIcon sx={{ mr: 2, color: 'primary.main' }} />
                <Box sx={{ flexGrow: 1 }}>
                  <Typography variant="h6">
                    Contract Review
                  </Typography>
                  <Box sx={{ display: 'flex', gap: 1, mt: 1, flexWrap: 'wrap' }}>
                    <Chip
                      label={`Status: ${review.status || 'Unknown'}`}
                      color={getStatusColor(review.status)}
                      size="small"
                      icon={getStatusIcon(review.status)}
                    />
                    {review.processing_time && (
                      <Chip
                        label={`Processing Time: ${formatProcessingTime(review.processing_time)}`}
                        variant="outlined"
                        size="small"
                        icon={<ScheduleIcon />}
                      />
                    )}
                    <Chip
                      label={review.review_type || 'Standard'}
                      color="secondary"
                      size="small"
                      variant="outlined"
                    />
                    <Chip
                      label={review.model_id || 'Unknown Model'}
                      variant="outlined"
                      size="small"
                    />
                    <Chip
                      label={formatDate(review.created_at)}
                      variant="outlined"
                      size="small"
                      icon={<ScheduleIcon />}
                    />
                  </Box>
                </Box>
                <Button
                  variant="outlined"
                  size="small"
                  startIcon={<ViewIcon />}
                  onClick={(e) => {
                    e.stopPropagation();
                    handleViewDetails('review', review.id);
                  }}
                  sx={{ ml: 2 }}
                >
                  View Details
                </Button>
              </Box>
            </AccordionSummary>
            <AccordionDetails>
              <Grid container spacing={2}>
                <Grid item xs={12} md={6}>
                  <Paper sx={{ p: 2, bgcolor: 'grey.50' }}>
                    <Typography variant="subtitle2" gutterBottom sx={{ fontWeight: 600 }}>
                      Contract Files
                    </Typography>
                    {review.contract_files && Array.isArray(review.contract_files) ? (
                      review.contract_files.map((file, fileIndex) => (
                        <Typography key={fileIndex} variant="body2">
                          <FileIcon sx={{ mr: 1, fontSize: 16 }} />
                          {file}
                        </Typography>
                      ))
                    ) : (
                      <Typography variant="body2" color="text.secondary">
                        No files specified
                      </Typography>
                    )}
                  </Paper>
                </Grid>
                <Grid item xs={12} md={6}>
                  <Paper sx={{ p: 2, bgcolor: 'grey.50' }}>
                    <Typography variant="subtitle2" gutterBottom sx={{ fontWeight: 600 }}>
                      Timeline
                    </Typography>
                    <Typography variant="body2">
                      <strong>Created:</strong> {formatDate(review.created_at)}
                    </Typography>
                    {review.completed_at && (
                      <Typography variant="body2">
                        <strong>Completed:</strong> {formatDate(review.completed_at)}
                      </Typography>
                    )}
                  </Paper>
                </Grid>
                {review.review_result && (
                  <Grid item xs={12}>
                    <Paper sx={{ p: 3, bgcolor: 'primary.light' }}>
                      <Typography variant="subtitle2" gutterBottom sx={{ fontWeight: 600, color: 'primary.dark' }}>
                        Review Result
                      </Typography>
                      <Box sx={{ maxHeight: '400px', overflow: 'auto' }}>
                        <Typography variant="body2" sx={{ whiteSpace: 'pre-wrap', lineHeight: 1.6, color: 'primary.dark' }}>
                          {typeof review.review_result === 'string' 
                            ? review.review_result 
                            : JSON.stringify(review.review_result, null, 2)
                          }
                        </Typography>
                      </Box>
                    </Paper>
                  </Grid>
                )}
              </Grid>
            </AccordionDetails>
          </Accordion>
        ))}
      </Box>
    );
  };

  const renderCommentsList = () => {
    const filteredComments = filterItems(comments, 'comments');
    
    if (filteredComments.length === 0) {
      return (
        <Paper sx={{ p: 4, textAlign: 'center' }}>
          <AIIcon sx={{ fontSize: 48, color: 'text.secondary', mb: 2 }} />
          <Typography variant="h6" color="text.secondary">
            No AI comments found
          </Typography>
          <Typography variant="body2" color="text.secondary">
            {searchTerm || statusFilter !== 'all' ? 'Try adjusting your filters' : 'Start by generating some AI comments'}
          </Typography>
        </Paper>
      );
    }

    return (
      <Box>
        {filteredComments.map((comment, index) => {
          const commentsResult = comment.comments_result;
          const hasNewFormat = commentsResult && (commentsResult.comments || commentsResult.document_comparisons);
          
          return (
            <Accordion 
              key={comment.id || index}
              sx={{ mb: 2 }}
            >
              <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                <Box sx={{ display: 'flex', alignItems: 'center', width: '100%' }}>
                  <AIIcon sx={{ mr: 2, color: 'purple' }} />
                  <Box sx={{ flexGrow: 1 }}>
                    <Typography variant="h6">
                      AI Comments & Analysis
                    </Typography>
                    <Box sx={{ display: 'flex', gap: 1, mt: 1, flexWrap: 'wrap' }}>
                      {comment.law_firm_name && (
                        <Chip
                          label={comment.law_firm_name}
                          icon={<BusinessIcon />}
                          size="small"
                          variant="outlined"
                        />
                      )}
                      {/* Display contract files for batch comments */}
                      {hasNewFormat && commentsResult.contract_files ? (
                        <Chip
                          label={`${commentsResult.contract_files.length} files`}
                          icon={<FileIcon />}
                          size="small"
                          color="primary"
                          variant="outlined"
                        />
                      ) : comment.contract_filename && (
                        <Chip
                          label={comment.contract_filename}
                          icon={<FileIcon />}
                          size="small"
                          color="primary"
                          variant="outlined"
                        />
                      )}
                      <Chip
                        label={`Status: ${comment.status || 'Unknown'}`}
                        color={getStatusColor(comment.status)}
                        size="small"
                        icon={getStatusIcon(comment.status)}
                      />
                      {comment.processing_time && (
                        <Chip
                          label={`Processing Time: ${formatProcessingTime(comment.processing_time)}`}
                          variant="outlined"
                          size="small"
                          icon={<ScheduleIcon />}
                        />
                      )}
                      <Chip
                        label={formatDate(comment.created_at)}
                        variant="outlined"
                        size="small"
                        icon={<ScheduleIcon />}
                      />
                    </Box>
                  </Box>
                  <Button
                    variant="outlined"
                    size="small"
                    startIcon={<ViewIcon />}
                    onClick={(e) => {
                      e.stopPropagation();
                      handleViewDetails('comment', comment.id);
                    }}
                    sx={{ ml: 2 }}
                  >
                    View Details
                  </Button>
                </Box>
              </AccordionSummary>
              <AccordionDetails>
                <Grid container spacing={2}>
                  <Grid item xs={12} md={6}>
                    <Paper sx={{ p: 2, bgcolor: 'grey.50' }}>
                      <Typography variant="subtitle2" gutterBottom sx={{ fontWeight: 600 }}>
                        Contract Information
                      </Typography>
                      {hasNewFormat && commentsResult.contract_files ? (
                        <>
                          <Typography variant="body2">
                            <strong>Files:</strong> {commentsResult.contract_files.length} documents
                          </Typography>
                          {commentsResult.contract_files.map((file, fileIndex) => (
                            <Typography key={fileIndex} variant="body2" sx={{ ml: 2 }}>
                              • {file.filename} {file.document_type && `(${file.document_type})`}
                            </Typography>
                          ))}
                        </>
                      ) : (
                        <Typography variant="body2">
                          <strong>File:</strong> {comment.contract_filename || 'N/A'}
                        </Typography>
                      )}
                      <Typography variant="body2">
                        <strong>Law Firm:</strong> {comment.law_firm_name || 'N/A'}
                      </Typography>
                    </Paper>
                  </Grid>
                  <Grid item xs={12} md={6}>
                    <Paper sx={{ p: 2, bgcolor: 'grey.50' }}>
                      <Typography variant="subtitle2" gutterBottom sx={{ fontWeight: 600 }}>
                        Timeline
                      </Typography>
                      <Typography variant="body2">
                        <strong>Created:</strong> {formatDate(comment.created_at)}
                      </Typography>
                      {comment.completed_at && (
                        <Typography variant="body2">
                          <strong>Completed:</strong> {formatDate(comment.completed_at)}
                        </Typography>
                      )}
                    </Paper>
                  </Grid>
                  
                  {/* Display new format with separate sections */}
                  {hasNewFormat ? (
                    <>
                      {/* Legal Comments Section */}
                      {commentsResult.comments && commentsResult.comments.length > 0 && (
                        <Grid item xs={12}>
                          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mt: 2, mb: 2 }}>
                            <Typography variant="subtitle2" sx={{ fontWeight: 600 }}>
                              Legal Comments ({commentsResult.comments.length})
                            </Typography>
                            
                            {/* View Mode Toggle */}
                            <ToggleButtonGroup
                              value={commentsViewMode}
                              exclusive
                              onChange={(event, newMode) => {
                                if (newMode !== null) {
                                  setCommentsViewMode(newMode);
                                }
                              }}
                              size="small"
                            >
                              <ToggleButton value="list">
                                <ListViewIcon sx={{ mr: 1 }} />
                                List View
                              </ToggleButton>
                              <ToggleButton value="compact">
                                <CompactViewIcon sx={{ mr: 1 }} />
                                Compact View
                              </ToggleButton>
                            </ToggleButtonGroup>
                          </Box>
                          
                          {/* List View */}
                          {commentsViewMode === 'list' && (
                            <Paper sx={{ p: 2, bgcolor: 'secondary.light', maxHeight: '400px', overflow: 'auto' }}>
                              <Grid container spacing={2}>
                                {commentsResult.comments.map((commentItem, commentIndex) => {
                                  const SeverityIcon = COMMENT_SEVERITY_ICONS[commentItem.severity] || InfoIcon;
                                  return (
                                    <Grid item xs={12} key={commentIndex}>
                                      <Card variant="outlined" sx={{ bgcolor: `${COMMENT_SEVERITY_COLORS[commentItem.severity]}.light` }}>
                                        <CardContent>
                                          <Box sx={{ display: 'flex', alignItems: 'flex-start', mb: 2 }}>
                                            <SeverityIcon sx={{ mr: 2, mt: 0.5, color: `${COMMENT_SEVERITY_COLORS[commentItem.severity]}.dark` }} />
                                            <Box sx={{ flexGrow: 1 }}>
                                              <Typography variant="subtitle2" sx={{ fontWeight: 600, color: `${COMMENT_SEVERITY_COLORS[commentItem.severity]}.dark` }}>
                                                {commentItem.section || 'General Comment'}
                                              </Typography>
                                              <Typography variant="body2" sx={{ mt: 1, color: `${COMMENT_SEVERITY_COLORS[commentItem.severity]}.dark` }}>
                                                {commentItem.comment}
                                              </Typography>
                                              {commentItem.suggestion && (
                                                <Typography variant="body2" sx={{ mt: 1, fontStyle: 'italic', color: `${COMMENT_SEVERITY_COLORS[commentItem.severity]}.dark` }}>
                                                  <strong>Suggestion:</strong> {commentItem.suggestion}
                                                </Typography>
                                              )}
                                            </Box>
                                            <Chip
                                              label={commentItem.severity?.toUpperCase() || 'MEDIUM'}
                                              size="small"
                                              color={COMMENT_SEVERITY_COLORS[commentItem.severity]}
                                              sx={{ ml: 2 }}
                                            />
                                          </Box>
                                        </CardContent>
                                      </Card>
                                    </Grid>
                                  );
                                })}
                              </Grid>
                            </Paper>
                          )}

                          {/* Compact View */}
                          {commentsViewMode === 'compact' && (
                            <Paper sx={{ p: 3, bgcolor: 'grey.50', maxHeight: '400px', overflow: 'auto' }}>
                              <Typography variant="body1" sx={{ whiteSpace: 'pre-wrap', lineHeight: 1.8, fontFamily: 'monospace' }}>
                                {commentsResult.comments.map((commentItem, commentIndex) => {
                                  const severityLabel = (commentItem.severity || 'medium').toUpperCase();
                                  const section = commentItem.section || 'GENERAL';
                                  return `[${severityLabel}] ${section}\n${commentItem.comment}${commentItem.suggestion ? `\n→ Suggestion: ${commentItem.suggestion}` : ''}\n\n`;
                                }).join('')}
                              </Typography>
                            </Paper>
                          )}
                        </Grid>
                      )}
                      
                      {/* Document Comparisons Section */}
                      {commentsResult.document_comparisons && commentsResult.document_comparisons.length > 0 && (
                        <Grid item xs={12}>
                          <Paper sx={{ p: 3, bgcolor: 'primary.light' }}>
                            <Typography variant="subtitle2" gutterBottom sx={{ fontWeight: 600, color: 'primary.dark' }}>
                              Document Comparisons ({commentsResult.document_comparisons.length})
                            </Typography>
                            <Box sx={{ maxHeight: '400px', overflow: 'auto' }}>
                              {commentsResult.document_comparisons.map((docComparison, docIndex) => (
                                <Box key={docIndex} sx={{ mb: 3, p: 2, bgcolor: 'white', borderRadius: 1 }}>
                                  <Typography variant="subtitle2" sx={{ fontWeight: 600, mb: 2, display: 'flex', alignItems: 'center' }}>
                                    <FileIcon sx={{ mr: 1 }} />
                                    {docComparison.contract_filename}
                                    {docComparison.document_type && (
                                      <Chip 
                                        label={docComparison.document_type} 
                                        size="small" 
                                        sx={{ ml: 1 }} 
                                      />
                                    )}
                                  </Typography>
                                  
                                  {docComparison.comparisons && docComparison.comparisons.map((comparison, compIndex) => (
                                    <Box key={compIndex} sx={{ mb: 2, p: 2, bgcolor: 'grey.50', borderRadius: 1 }}>
                                      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', mb: 1 }}>
                                        <Box sx={{ flexGrow: 1 }}>
                                          <Typography variant="body2" sx={{ fontWeight: 600, mb: 1 }}>
                                            vs {comparison.template_filename}
                                          </Typography>
                                          <Box sx={{ display: 'flex', gap: 1, mb: 1 }}>
                                            <Chip
                                              label={`${comparison.differences ? comparison.differences.length : 0} differences`}
                                              size="small"
                                              color={comparison.differences?.length > 0 ? 'warning' : 'success'}
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
                                      
                                      {comparison.differences && comparison.differences.length > 0 && (
                                        <Box sx={{ mt: 1, maxHeight: '200px', overflow: 'auto' }}>
                                          {comparison.differences.slice(0, 5).map((diff, diffIndex) => (
                                            <Typography 
                                              key={diffIndex} 
                                              variant="caption" 
                                              sx={{ 
                                                display: 'block',
                                                p: 0.5,
                                                bgcolor: diff.type === 'added' ? 'success.light' : 'error.light',
                                                color: diff.type === 'added' ? 'success.dark' : 'error.dark',
                                                borderRadius: 0.5,
                                                mb: 0.5,
                                                fontFamily: 'monospace'
                                              }}
                                            >
                                              {diff.type === 'added' ? '+' : '-'} {diff.content.substring(0, 100)}
                                              {diff.content.length > 100 && '...'}
                                            </Typography>
                                          ))}
                                          {comparison.differences.length > 5 && (
                                            <Typography variant="caption" color="text.secondary">
                                              ... and {comparison.differences.length - 5} more differences
                                            </Typography>
                                          )}
                                        </Box>
                                      )}
                                    </Box>
                                  ))}
                                </Box>
                              ))}
                            </Box>
                          </Paper>
                        </Grid>
                      )}
                    </>
                  ) : (
                    /* Legacy format display */
                    comment.comments_result && (
                      <Grid item xs={12}>
                        <Paper sx={{ p: 3, bgcolor: 'secondary.light' }}>
                          <Typography variant="subtitle2" gutterBottom sx={{ fontWeight: 600, color: 'secondary.dark' }}>
                            AI Comments & Analysis
                          </Typography>
                          <Box sx={{ maxHeight: '400px', overflow: 'auto' }}>
                            <Typography variant="body2" sx={{ whiteSpace: 'pre-wrap', lineHeight: 1.6, color: 'secondary.dark' }}>
                              {typeof comment.comments_result === 'string' 
                                ? comment.comments_result 
                                : JSON.stringify(comment.comments_result, null, 2)
                              }
                            </Typography>
                          </Box>
                        </Paper>
                      </Grid>
                    )
                  )}
                </Grid>
              </AccordionDetails>
            </Accordion>
          );
        })}
      </Box>
    );
  };

  const renderDetailDialog = () => {
    if (!selectedDetail) return null;

    const { type, data } = selectedDetail;
    const commentsResult = type === 'comment' ? data.comments_result : null;
    const hasNewFormat = commentsResult && (commentsResult.comments || commentsResult.document_comparisons);

    return (
      <Dialog
        open={detailDialogOpen}
        onClose={handleCloseDialog}
        maxWidth="lg"
        fullWidth
      >
        <DialogTitle sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <Box sx={{ display: 'flex', alignItems: 'center' }}>
            {type === 'review' ? <ReviewIcon sx={{ mr: 1 }} /> : <AIIcon sx={{ mr: 1 }} />}
            {type === 'review' ? 'Contract Review Details' : 'AI Comments Details'}
          </Box>
          <Button onClick={handleCloseDialog} startIcon={<CloseIcon />}>
            Close
          </Button>
        </DialogTitle>
        <DialogContent>
          <Grid container spacing={2}>
            <Grid item xs={12} md={6}>
              <Typography variant="subtitle2" gutterBottom sx={{ fontWeight: 600 }}>
                Basic Information
              </Typography>
              <Typography variant="body2"><strong>ID:</strong> {data.id || 'N/A'}</Typography>
              <Typography variant="body2"><strong>Status:</strong> {data.status || 'N/A'}</Typography>
              <Typography variant="body2"><strong>Created:</strong> {formatDate(data.created_at)}</Typography>
              {data.completed_at && (
                <Typography variant="body2"><strong>Completed:</strong> {formatDate(data.completed_at)}</Typography>
              )}
              {data.processing_time && (
                <Typography variant="body2"><strong>Processing Time:</strong> {formatProcessingTime(data.processing_time)}</Typography>
              )}
            </Grid>
            <Grid item xs={12} md={6}>
              <Typography variant="subtitle2" gutterBottom sx={{ fontWeight: 600 }}>
                {type === 'review' ? 'Review Information' : 'Comment Information'}
              </Typography>
              {type === 'review' ? (
                <>
                  <Typography variant="body2"><strong>Review Type:</strong> {data.review_type || 'N/A'}</Typography>
                  <Typography variant="body2"><strong>Model ID:</strong> {data.model_id || 'N/A'}</Typography>
                  <Typography variant="body2"><strong>Contract Files:</strong> {
                    data.contract_files && Array.isArray(data.contract_files) 
                      ? data.contract_files.join(', ') 
                      : 'N/A'
                  }</Typography>
                </>
              ) : (
                <>
                  {hasNewFormat && commentsResult.contract_files ? (
                    <>
                      <Typography variant="body2"><strong>Contract Files:</strong> {commentsResult.contract_files.length} documents</Typography>
                      {commentsResult.contract_files.map((file, fileIndex) => (
                        <Typography key={fileIndex} variant="body2" sx={{ ml: 2 }}>
                          • {file.filename} {file.document_type && `(${file.document_type})`}
                        </Typography>
                      ))}
                    </>
                  ) : (
                    <Typography variant="body2"><strong>Contract File:</strong> {data.contract_filename || 'N/A'}</Typography>
                  )}
                  <Typography variant="body2"><strong>Law Firm:</strong> {data.law_firm_name || 'N/A'}</Typography>
                </>
              )}
            </Grid>
            
            {/* Display new format with separate sections */}
            {type === 'comment' && hasNewFormat ? (
              <>
                {/* Legal Comments Section */}
                {commentsResult.comments && commentsResult.comments.length > 0 && (
                  <Grid item xs={12}>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mt: 2, mb: 2 }}>
                      <Typography variant="subtitle2" sx={{ fontWeight: 600 }}>
                        Legal Comments ({commentsResult.comments.length})
                      </Typography>
                      
                      {/* View Mode Toggle */}
                      <ToggleButtonGroup
                        value={commentsViewMode}
                        exclusive
                        onChange={(event, newMode) => {
                          if (newMode !== null) {
                            setCommentsViewMode(newMode);
                          }
                        }}
                        size="small"
                      >
                        <ToggleButton value="list">
                          <ListViewIcon sx={{ mr: 1 }} />
                          List View
                        </ToggleButton>
                        <ToggleButton value="compact">
                          <CompactViewIcon sx={{ mr: 1 }} />
                          Compact View
                        </ToggleButton>
                      </ToggleButtonGroup>
                    </Box>
                    
                    {/* List View */}
                    {commentsViewMode === 'list' && (
                      <Paper sx={{ p: 2, bgcolor: 'secondary.light', maxHeight: '400px', overflow: 'auto' }}>
                        <Grid container spacing={2}>
                          {commentsResult.comments.map((commentItem, commentIndex) => {
                            const SeverityIcon = COMMENT_SEVERITY_ICONS[commentItem.severity] || InfoIcon;
                            return (
                              <Grid item xs={12} key={commentIndex}>
                                <Card variant="outlined" sx={{ bgcolor: `${COMMENT_SEVERITY_COLORS[commentItem.severity]}.light` }}>
                                  <CardContent>
                                    <Box sx={{ display: 'flex', alignItems: 'flex-start', mb: 2 }}>
                                      <SeverityIcon sx={{ mr: 2, mt: 0.5, color: `${COMMENT_SEVERITY_COLORS[commentItem.severity]}.dark` }} />
                                      <Box sx={{ flexGrow: 1 }}>
                                        <Typography variant="subtitle2" sx={{ fontWeight: 600, color: `${COMMENT_SEVERITY_COLORS[commentItem.severity]}.dark` }}>
                                          {commentItem.section || 'General Comment'}
                                        </Typography>
                                        <Typography variant="body2" sx={{ mt: 1, color: `${COMMENT_SEVERITY_COLORS[commentItem.severity]}.dark` }}>
                                          {commentItem.comment}
                                        </Typography>
                                        {commentItem.suggestion && (
                                          <Typography variant="body2" sx={{ mt: 1, fontStyle: 'italic', color: `${COMMENT_SEVERITY_COLORS[commentItem.severity]}.dark` }}>
                                            <strong>Suggestion:</strong> {commentItem.suggestion}
                                          </Typography>
                                        )}
                                      </Box>
                                      <Chip
                                        label={commentItem.severity?.toUpperCase() || 'MEDIUM'}
                                        size="small"
                                        color={COMMENT_SEVERITY_COLORS[commentItem.severity]}
                                        sx={{ ml: 2 }}
                                      />
                                    </Box>
                                  </CardContent>
                                </Card>
                              </Grid>
                            );
                          })}
                        </Grid>
                      </Paper>
                    )}

                    {/* Compact View */}
                    {commentsViewMode === 'compact' && (
                      <Paper sx={{ p: 3, bgcolor: 'grey.50', maxHeight: '400px', overflow: 'auto' }}>
                        <Typography variant="body1" sx={{ whiteSpace: 'pre-wrap', lineHeight: 1.8, fontFamily: 'monospace' }}>
                          {commentsResult.comments.map((commentItem, commentIndex) => {
                            const severityLabel = (commentItem.severity || 'medium').toUpperCase();
                            const section = commentItem.section || 'GENERAL';
                            return `[${severityLabel}] ${section}\n${commentItem.comment}${commentItem.suggestion ? `\n→ Suggestion: ${commentItem.suggestion}` : ''}\n\n`;
                          }).join('')}
                        </Typography>
                      </Paper>
                    )}
                  </Grid>
                )}
                
                {/* Document Comparisons Section */}
                {commentsResult.document_comparisons && commentsResult.document_comparisons.length > 0 && (
                  <Grid item xs={12}>
                    <Paper sx={{ p: 3, bgcolor: 'primary.light' }}>
                      <Typography variant="subtitle2" gutterBottom sx={{ fontWeight: 600, color: 'primary.dark' }}>
                        Document Comparisons ({commentsResult.document_comparisons.length})
                      </Typography>
                      <Box sx={{ maxHeight: '400px', overflow: 'auto' }}>
                        {commentsResult.document_comparisons.map((docComparison, docIndex) => (
                          <Box key={docIndex} sx={{ mb: 3, p: 2, bgcolor: 'white', borderRadius: 1 }}>
                            <Typography variant="subtitle2" sx={{ fontWeight: 600, mb: 2 }}>
                              {docComparison.contract_filename}
                              {docComparison.document_type && (
                                <Chip 
                                  label={docComparison.document_type} 
                                  size="small" 
                                  sx={{ ml: 1 }} 
                                />
                              )}
                            </Typography>
                            
                            {docComparison.comparisons && docComparison.comparisons.map((comparison, compIndex) => (
                              <Box key={compIndex} sx={{ mb: 2, p: 2, bgcolor: 'grey.50', borderRadius: 1 }}>
                                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', mb: 1 }}>
                                  <Box sx={{ flexGrow: 1 }}>
                                    <Typography variant="body2" sx={{ fontWeight: 600, mb: 1 }}>
                                      vs {comparison.template_filename}
                                    </Typography>
                                    <Box sx={{ display: 'flex', gap: 1, mb: 1 }}>
                                      <Chip
                                        label={`${comparison.differences ? comparison.differences.length : 0} differences`}
                                        size="small"
                                        color={comparison.differences?.length > 0 ? 'warning' : 'success'}
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
                                
                                {comparison.differences && comparison.differences.length > 0 && (
                                  <Box sx={{ mt: 1, maxHeight: '200px', overflow: 'auto' }}>
                                    {comparison.differences.slice(0, 5).map((diff, diffIndex) => (
                                      <Typography 
                                        key={diffIndex} 
                                        variant="caption" 
                                        sx={{ 
                                          display: 'block',
                                          p: 0.5,
                                          bgcolor: diff.type === 'added' ? 'success.light' : 'error.light',
                                          color: diff.type === 'added' ? 'success.dark' : 'error.dark',
                                          borderRadius: 0.5,
                                          mb: 0.5,
                                          fontFamily: 'monospace'
                                        }}
                                      >
                                        {diff.type === 'added' ? '+' : '-'} {diff.content.substring(0, 100)}
                                        {diff.content.length > 100 && '...'}
                                      </Typography>
                                    ))}
                                    {comparison.differences.length > 5 && (
                                      <Typography variant="caption" color="text.secondary">
                                        ... and {comparison.differences.length - 5} more differences
                                      </Typography>
                                    )}
                                  </Box>
                                )}
                              </Box>
                            ))}
                          </Box>
                        ))}
                      </Box>
                    </Paper>
                  </Grid>
                )}
              </>
            ) : (
              /* Legacy format or review result display */
              (data.review_result || data.comments_result) && (
                <Grid item xs={12}>
                  <Typography variant="subtitle2" gutterBottom sx={{ fontWeight: 600, mt: 2 }}>
                    {type === 'review' ? 'Review Result' : 'Comments Result'}
                  </Typography>
                  <Paper sx={{ p: 2, bgcolor: 'grey.50', maxHeight: '400px', overflow: 'auto' }}>
                    <Typography variant="body2" sx={{ whiteSpace: 'pre-wrap', fontFamily: 'monospace' }}>
                      {typeof (data.review_result || data.comments_result) === 'string' 
                        ? (data.review_result || data.comments_result)
                        : JSON.stringify((data.review_result || data.comments_result), null, 2)
                      }
                    </Typography>
                  </Paper>
                </Grid>
              )
            )}
          </Grid>
        </DialogContent>
        <DialogActions>
          <Button onClick={handleCloseDialog}>Close</Button>
        </DialogActions>
      </Dialog>
    );
  };

  if (reviewsError || commentsError) {
    return (
      <Box sx={{ p: 3 }}>
        <Alert severity="error" sx={{ mb: 2 }}>
          Error loading data: {reviewsError?.message || commentsError?.message}
        </Alert>
        <Button 
          variant="contained" 
          onClick={() => {
            refetchReviews();
            refetchComments();
          }}
          startIcon={<RefreshIcon />}
        >
          Retry
        </Button>
      </Box>
    );
  }

  return (
    <Box sx={{ p: 3 }}>
      <Typography variant="h4" gutterBottom>
        Review History
      </Typography>
      
      {/* Filters */}
      <Paper sx={{ p: 2, mb: 3 }}>
        <Grid container spacing={2} alignItems="center">
          <Grid item xs={12} md={4}>
            <TextField
              fullWidth
              label="Search"
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              InputProps={{
                startAdornment: <SearchIcon sx={{ mr: 1, color: 'text.secondary' }} />
              }}
            />
          </Grid>
          <Grid item xs={12} md={3}>
            <FormControl fullWidth>
              <InputLabel>Status Filter</InputLabel>
              <Select
                value={statusFilter}
                label="Status Filter"
                onChange={(e) => setStatusFilter(e.target.value)}
                startAdornment={<FilterIcon sx={{ mr: 1, color: 'text.secondary' }} />}
              >
                <MenuItem value="all">All Statuses</MenuItem>
                <MenuItem value="completed">Completed</MenuItem>
                <MenuItem value="processing">Processing</MenuItem>
                <MenuItem value="failed">Failed</MenuItem>
              </Select>
            </FormControl>
          </Grid>
          <Grid item xs={12} md={2}>
            <Button
              fullWidth
              variant="outlined"
              onClick={() => {
                refetchReviews();
                refetchComments();
              }}
              startIcon={<RefreshIcon />}
            >
              Refresh
            </Button>
          </Grid>
        </Grid>
      </Paper>

      {/* Tabs */}
      <Paper sx={{ mb: 3 }}>
        <Tabs
          value={activeTab}
          onChange={(event, newValue) => setActiveTab(newValue)}
          variant="fullWidth"
        >
          <Tab 
            label={
              <Badge badgeContent={reviews.length} color="primary">
                Contract Reviews
              </Badge>
            } 
          />
          <Tab 
            label={
              <Badge badgeContent={comments.length} color="secondary">
                AI Comments
              </Badge>
            } 
          />
        </Tabs>
      </Paper>

      {/* Content */}
      {reviewsLoading || commentsLoading ? (
        <Box sx={{ display: 'flex', justifyContent: 'center', p: 4 }}>
          <CircularProgress />
        </Box>
      ) : (
        <>
          {activeTab === 0 && renderReviewsList()}
          {activeTab === 1 && renderCommentsList()}
        </>
      )}

      {/* Detail Dialog */}
      {renderDetailDialog()}
      
      {/* Diff Viewer Dialog */}
      {diffViewOpen && selectedComparison && (
        <DiffViewer
          differences={selectedComparison.differences}
          templateFilename={selectedComparison.template_filename}
          contractFilename={selectedComparison.contract_filename}
          onClose={() => {
            setDiffViewOpen(false);
            setSelectedComparison(null);
          }}
        />
      )}
    </Box>
  );
};

export default ReviewHistory; 