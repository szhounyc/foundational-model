import React, { useState } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Grid,
  Chip,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  Paper,
  LinearProgress,
  Button,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Divider,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  TextField,
  InputAdornment,
} from '@mui/material';
import {
  Description as DocumentIcon,
  CheckCircle as CheckIcon,
  Error as ErrorIcon,
  Schedule as ScheduleIcon,
  Visibility as ViewIcon,
  Search as SearchIcon,
  FilterList as FilterIcon,
} from '@mui/icons-material';
import { useQuery } from 'react-query';
import axios from 'axios';

const API_BASE = '';

function ReviewHistory() {
  const [selectedReview, setSelectedReview] = useState(null);
  const [dialogOpen, setDialogOpen] = useState(false);
  const [statusFilter, setStatusFilter] = useState('all');
  const [searchTerm, setSearchTerm] = useState('');

  // Get all reviews
  const { data: reviewsData, isLoading: reviewsLoading } = useQuery(
    'all-reviews',
    async () => {
      const response = await axios.get(`${API_BASE}/api/contracts/reviews`);
      return response.data;
    },
    {
      refetchInterval: 30000,
    }
  );

  const reviews = reviewsData?.reviews || [];

  // Filter reviews based on status and search term
  const filteredReviews = reviews.filter(review => {
    const matchesStatus = statusFilter === 'all' || review.status === statusFilter;
    const matchesSearch = searchTerm === '' || 
      review.id.toLowerCase().includes(searchTerm.toLowerCase()) ||
      review.model_id.toLowerCase().includes(searchTerm.toLowerCase()) ||
      review.review_type.toLowerCase().includes(searchTerm.toLowerCase());
    
    return matchesStatus && matchesSearch;
  });

  const formatDate = (dateString) => {
    return new Date(dateString).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    });
  };

  const getStatusColor = (status) => {
    switch (status) {
      case 'completed':
        return 'success';
      case 'processing':
        return 'warning';
      case 'failed':
        return 'error';
      default:
        return 'default';
    }
  };

  const getStatusIcon = (status) => {
    switch (status) {
      case 'completed':
        return <CheckIcon />;
      case 'processing':
        return <ScheduleIcon />;
      case 'failed':
        return <ErrorIcon />;
      default:
        return <DocumentIcon />;
    }
  };

  const handleViewReview = (review) => {
    setSelectedReview(review);
    setDialogOpen(true);
  };

  const handleCloseDialog = () => {
    setDialogOpen(false);
    setSelectedReview(null);
  };

  // Statistics
  const totalReviews = reviews.length;
  const completedReviews = reviews.filter(r => r.status === 'completed').length;
  const processingReviews = reviews.filter(r => r.status === 'processing').length;
  const failedReviews = reviews.filter(r => r.status === 'failed').length;

  const stats = [
    { label: 'Total Reviews', value: totalReviews, color: 'primary' },
    { label: 'Completed', value: completedReviews, color: 'success' },
    { label: 'Processing', value: processingReviews, color: 'warning' },
    { label: 'Failed', value: failedReviews, color: 'error' },
  ];

  return (
    <Box sx={{ maxWidth: 1200, mx: 'auto' }}>
      <Typography variant="h4" gutterBottom sx={{ mb: 3, fontWeight: 600 }}>
        Review History
      </Typography>

      {/* Statistics */}
      <Grid container spacing={3} sx={{ mb: 4 }}>
        {stats.map((stat, index) => (
          <Grid item xs={12} sm={6} md={3} key={index}>
            <Card>
              <CardContent sx={{ textAlign: 'center' }}>
                <Typography variant="h3" color={`${stat.color}.main`} sx={{ fontWeight: 600 }}>
                  {stat.value}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  {stat.label}
                </Typography>
              </CardContent>
            </Card>
          </Grid>
        ))}
      </Grid>

      {/* Filters */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Grid container spacing={2} alignItems="center">
            <Grid item xs={12} md={4}>
              <TextField
                fullWidth
                placeholder="Search reviews..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                InputProps={{
                  startAdornment: (
                    <InputAdornment position="start">
                      <SearchIcon />
                    </InputAdornment>
                  ),
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
                  startAdornment={<FilterIcon sx={{ mr: 1 }} />}
                >
                  <MenuItem value="all">All Statuses</MenuItem>
                  <MenuItem value="completed">Completed</MenuItem>
                  <MenuItem value="processing">Processing</MenuItem>
                  <MenuItem value="failed">Failed</MenuItem>
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={12} md={5}>
              <Typography variant="body2" color="text.secondary">
                Showing {filteredReviews.length} of {totalReviews} reviews
              </Typography>
            </Grid>
          </Grid>
        </CardContent>
      </Card>

      {/* Reviews List */}
      <Card>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Contract Reviews
          </Typography>

          {reviewsLoading ? (
            <Box sx={{ mt: 2 }}>
              <LinearProgress />
              <Typography variant="body2" sx={{ mt: 1 }}>
                Loading reviews...
              </Typography>
            </Box>
          ) : filteredReviews.length > 0 ? (
            <List>
              {filteredReviews.map((review, index) => (
                <ListItem
                  key={review.id}
                  sx={{
                    border: 1,
                    borderColor: 'grey.200',
                    borderRadius: 1,
                    mb: 1,
                    '&:hover': {
                      backgroundColor: 'grey.50',
                    },
                  }}
                >
                  <ListItemIcon>
                    {getStatusIcon(review.status)}
                  </ListItemIcon>
                  <ListItemText
                    primary={
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 0.5 }}>
                        <Typography variant="subtitle1" sx={{ fontWeight: 500 }}>
                          Review #{review.id.slice(0, 8)}
                        </Typography>
                        <Chip
                          size="small"
                          label={review.status}
                          color={getStatusColor(review.status)}
                        />
                        <Chip
                          size="small"
                          label={review.review_type}
                          variant="outlined"
                        />
                      </Box>
                    }
                    secondary={
                      <Box>
                        <Typography variant="body2" color="text.secondary">
                          Created: {formatDate(review.created_at)}
                          {review.completed_at && (
                            <> • Completed: {formatDate(review.completed_at)}</>
                          )}
                        </Typography>
                        <Typography variant="body2" color="text.secondary">
                          Model: {review.model_id} • Files: {review.contract_files?.length || 0}
                          {review.processing_time && (
                            <> • Processing: {review.processing_time.toFixed(1)}s</>
                          )}
                        </Typography>
                      </Box>
                    }
                  />
                  <Button
                    variant="outlined"
                    size="small"
                    startIcon={<ViewIcon />}
                    onClick={() => handleViewReview(review)}
                    disabled={review.status !== 'completed'}
                  >
                    View
                  </Button>
                </ListItem>
              ))}
            </List>
          ) : (
            <Paper sx={{ p: 3, textAlign: 'center', backgroundColor: 'grey.50' }}>
              <DocumentIcon sx={{ fontSize: 48, color: 'grey.400', mb: 2 }} />
              <Typography variant="h6" color="text.secondary" gutterBottom>
                {searchTerm || statusFilter !== 'all' ? 'No matching reviews found' : 'No reviews yet'}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                {searchTerm || statusFilter !== 'all' 
                  ? 'Try adjusting your search or filter criteria'
                  : 'Upload your first contract to get started'
                }
              </Typography>
            </Paper>
          )}
        </CardContent>
      </Card>

      {/* Review Details Dialog */}
      <Dialog
        open={dialogOpen}
        onClose={handleCloseDialog}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <Typography variant="h6">
              Review Details #{selectedReview?.id?.slice(0, 8)}
            </Typography>
            <Chip
              size="small"
              label={selectedReview?.status}
              color={getStatusColor(selectedReview?.status)}
            />
          </Box>
        </DialogTitle>
        <DialogContent>
          {selectedReview && (
            <Box>
              {/* Review Metadata */}
              <Grid container spacing={2} sx={{ mb: 3 }}>
                <Grid item xs={12} sm={6}>
                  <Typography variant="body2" color="text.secondary">
                    Review Type
                  </Typography>
                  <Typography variant="body1">
                    {selectedReview.review_type}
                  </Typography>
                </Grid>
                <Grid item xs={12} sm={6}>
                  <Typography variant="body2" color="text.secondary">
                    Model Used
                  </Typography>
                  <Typography variant="body1">
                    {selectedReview.model_id}
                  </Typography>
                </Grid>
                <Grid item xs={12} sm={6}>
                  <Typography variant="body2" color="text.secondary">
                    Created
                  </Typography>
                  <Typography variant="body1">
                    {formatDate(selectedReview.created_at)}
                  </Typography>
                </Grid>
                <Grid item xs={12} sm={6}>
                  <Typography variant="body2" color="text.secondary">
                    Processing Time
                  </Typography>
                  <Typography variant="body1">
                    {selectedReview.processing_time?.toFixed(1)}s
                  </Typography>
                </Grid>
              </Grid>

              <Divider sx={{ my: 2 }} />

              {/* Review Content */}
              <Typography variant="h6" gutterBottom>
                Review Results
              </Typography>
              
              {selectedReview.review_result ? (
                <Paper sx={{ p: 3, backgroundColor: 'grey.50', maxHeight: 400, overflow: 'auto' }}>
                  <Typography variant="body1" sx={{ whiteSpace: 'pre-wrap', lineHeight: 1.6 }}>
                    {selectedReview.review_result.review || 'No review content available'}
                  </Typography>
                </Paper>
              ) : (
                <Paper sx={{ p: 3, textAlign: 'center', backgroundColor: 'grey.50' }}>
                  <Typography variant="body2" color="text.secondary">
                    No review results available
                  </Typography>
                </Paper>
              )}

              {/* Additional metadata if available */}
              {selectedReview.review_result && (
                <Box sx={{ mt: 2, display: 'flex', gap: 1, flexWrap: 'wrap' }}>
                  {selectedReview.review_result.sections_reviewed && (
                    <Chip
                      size="small"
                      label={`Sections: ${selectedReview.review_result.sections_reviewed}`}
                      variant="outlined"
                    />
                  )}
                  {selectedReview.review_result.tokens_generated && (
                    <Chip
                      size="small"
                      label={`Tokens: ${selectedReview.review_result.tokens_generated}`}
                      variant="outlined"
                    />
                  )}
                </Box>
              )}
            </Box>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={handleCloseDialog}>Close</Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
}

export default ReviewHistory; 