import React from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Grid,
  Button,
  Chip,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  Paper,
  LinearProgress,
} from '@mui/material';
import {
  Description as DocumentIcon,
  Gavel as ReviewIcon,
  TrendingUp as TrendIcon,
  Speed as SpeedIcon,
  CheckCircle as CheckIcon,
  Schedule as ScheduleIcon,
  CloudUpload as UploadIcon,
  SmartToy as AIIcon,
} from '@mui/icons-material';
import { useQuery } from 'react-query';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';

const API_BASE = process.env.REACT_APP_API_URL || 'http://localhost:9100';

function Dashboard() {
  const navigate = useNavigate();

  // Get recent reviews
  const { data: reviewsData, isLoading: reviewsLoading } = useQuery(
    'recent-reviews',
    async () => {
      const response = await axios.get(`${API_BASE}/api/contracts/reviews`);
      return response.data;
    },
    {
      refetchInterval: 30000,
    }
  );

  // Get recent comments
  const { data: commentsData, isLoading: commentsLoading } = useQuery(
    'recent-comments',
    async () => {
      const response = await axios.get(`${API_BASE}/api/contracts/comments`);
      return response.data;
    },
    {
      refetchInterval: 30000,
    }
  );

  // Get inference service status
  const { data: inferenceStatus } = useQuery(
    'inference-status',
    async () => {
      const response = await axios.get(`${API_BASE}/inference/health`);
      return response.data;
    },
    {
      refetchInterval: 10000,
    }
  );

  const recentReviews = reviewsData?.reviews?.slice(0, 3) || [];
  const recentComments = (commentsData?.comments || (Array.isArray(commentsData) ? commentsData : [])).slice(0, 3) || [];
  const totalReviews = reviewsData?.reviews?.length || 0;
  const totalComments = commentsData?.comments?.length || (Array.isArray(commentsData) ? commentsData.length : 0) || 0;
  const completedReviews = reviewsData?.reviews?.filter(r => r.status === 'completed')?.length || 0;
  const completedComments = (commentsData?.comments || (Array.isArray(commentsData) ? commentsData : [])).filter(c => c.status === 'completed')?.length || 0;

  // Combine and sort recent activity
  const recentActivity = [
    ...recentReviews.map(review => ({ ...review, type: 'review' })),
    ...recentComments.map(comment => ({ ...comment, type: 'comment' }))
  ].sort((a, b) => new Date(b.created_at) - new Date(a.created_at)).slice(0, 3);

  const stats = [
    {
      title: 'Total Reviews',
      value: totalReviews,
      icon: <DocumentIcon />,
      color: 'primary',
    },
    {
      title: 'AI Comments',
      value: totalComments,
      icon: <AIIcon />,
      color: 'secondary',
    },
    {
      title: 'Success Rate',
      value: (totalReviews + totalComments) > 0 ? `${Math.round(((completedReviews + completedComments) / (totalReviews + totalComments)) * 100)}%` : '0%',
      icon: <TrendIcon />,
      color: 'info',
    },
    {
      title: 'AI Models',
      value: inferenceStatus?.model_count || 0,
      icon: <SpeedIcon />,
      color: 'warning',
    },
  ];

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

  const getActivityIcon = (type) => {
    return type === 'review' ? <DocumentIcon color="primary" /> : <AIIcon color="secondary" />;
  };

  const getActivityTitle = (item) => {
    if (item.type === 'review') {
      return `Review #${item.id}`;
    } else {
      // Check if it's a batch comment
      const commentsResult = item.comments_result;
      const hasNewFormat = commentsResult && (commentsResult.comments || commentsResult.document_comparisons);
      if (hasNewFormat && commentsResult.contract_files) {
        return `AI Comments (${commentsResult.contract_files.length} files)`;
      }
      return `AI Comments - ${item.contract_filename || 'Contract'}`;
    }
  };

  const getActivitySubtitle = (item) => {
    if (item.type === 'review') {
      return `${formatDate(item.created_at)} • ${item.review_type} • ${item.model_id}`;
    } else {
      return `${formatDate(item.created_at)} • ${item.law_firm_name || 'Unknown Firm'}`;
    }
  };

  return (
    <Box sx={{ maxWidth: 1200, mx: 'auto' }}>
      <Typography variant="h4" gutterBottom sx={{ mb: 3, fontWeight: 600 }}>
        Contract Review Dashboard
      </Typography>

      {/* Statistics Cards */}
      <Grid container spacing={3} sx={{ mb: 4 }}>
        {stats.map((stat, index) => (
          <Grid item xs={12} sm={6} md={3} key={index}>
            <Card>
              <CardContent>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                  <Box
                    sx={{
                      p: 1,
                      borderRadius: 1,
                      backgroundColor: `${stat.color}.light`,
                      color: `${stat.color}.main`,
                      mr: 2,
                    }}
                  >
                    {stat.icon}
                  </Box>
                  <Typography variant="h4" sx={{ fontWeight: 600 }}>
                    {stat.value}
                  </Typography>
                </Box>
                <Typography variant="body2" color="text.secondary">
                  {stat.title}
                </Typography>
              </CardContent>
            </Card>
          </Grid>
        ))}
      </Grid>

      <Grid container spacing={3}>
        {/* Quick Actions */}
        <Grid item xs={12} md={4}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Quick Actions
              </Typography>
              
              <Button
                variant="contained"
                size="large"
                fullWidth
                startIcon={<UploadIcon />}
                onClick={() => navigate('/contract-review')}
                sx={{ mb: 2, py: 1.5 }}
              >
                Upload & Review Contract
              </Button>

              <Button
                variant="outlined"
                size="large"
                fullWidth
                startIcon={<ReviewIcon />}
                onClick={() => navigate('/review-history')}
                sx={{ py: 1.5 }}
              >
                View Review History
              </Button>
            </CardContent>
          </Card>

          {/* Service Status */}
          <Card sx={{ mt: 3 }}>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Service Status
              </Typography>
              
              <Box sx={{ mb: 2 }}>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                  <Typography variant="body2">Inference Service</Typography>
                  <Chip
                    size="small"
                    label={inferenceStatus?.status || 'Unknown'}
                    color={inferenceStatus?.status === 'healthy' ? 'success' : 'error'}
                  />
                </Box>
                
                {inferenceStatus?.hardware && (
                  <Typography variant="caption" color="text.secondary">
                    {inferenceStatus.hardware.device} • {inferenceStatus.hardware.memory}
                  </Typography>
                )}
              </Box>

              {inferenceStatus?.loaded_models?.length > 0 && (
                <Box>
                  <Typography variant="body2" sx={{ mb: 1 }}>
                    Loaded Models ({inferenceStatus.loaded_models.length})
                  </Typography>
                  {inferenceStatus.loaded_models.map((model, index) => (
                    <Chip
                      key={index}
                      size="small"
                      label={model}
                      variant="outlined"
                      sx={{ mr: 0.5, mb: 0.5 }}
                    />
                  ))}
                </Box>
              )}
            </CardContent>
          </Card>
        </Grid>

        {/* Recent Activity */}
        <Grid item xs={12} md={8}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Recent Activity (Last 3)
              </Typography>

              {(reviewsLoading || commentsLoading) ? (
                <Box sx={{ mt: 2 }}>
                  <LinearProgress />
                  <Typography variant="body2" sx={{ mt: 1 }}>
                    Loading recent activity...
                  </Typography>
                </Box>
              ) : recentActivity.length > 0 ? (
                <List>
                  {recentActivity.map((item) => (
                    <ListItem
                      key={`${item.type}-${item.id}`}
                      sx={{
                        border: 1,
                        borderColor: 'grey.200',
                        borderRadius: 1,
                        mb: 1,
                      }}
                    >
                      <ListItemIcon>
                        {getActivityIcon(item.type)}
                      </ListItemIcon>
                      <ListItemText
                        primary={
                          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                            <Typography variant="subtitle2">
                              {getActivityTitle(item)}
                            </Typography>
                            <Chip
                              size="small"
                              label={item.status}
                              color={getStatusColor(item.status)}
                            />
                            <Chip
                              size="small"
                              label={item.type === 'review' ? 'Review' : 'AI Comment'}
                              variant="outlined"
                              color={item.type === 'review' ? 'primary' : 'secondary'}
                            />
                          </Box>
                        }
                        secondary={
                          <Box sx={{ mt: 0.5 }}>
                            <Typography variant="caption" color="text.secondary">
                              {getActivitySubtitle(item)}
                            </Typography>
                            {item.type === 'review' && item.contract_files?.length > 0 && (
                              <Typography variant="caption" sx={{ display: 'block' }}>
                                {item.contract_files.length} file(s) reviewed
                              </Typography>
                            )}
                            {item.type === 'comment' && item.processing_time && (
                              <Typography variant="caption" sx={{ display: 'block' }}>
                                Processing time: {parseFloat(item.processing_time).toFixed(2)}s
                              </Typography>
                            )}
                          </Box>
                        }
                      />
                      {item.status === 'processing' && (
                        <Box sx={{ ml: 2 }}>
                          <ScheduleIcon color="warning" />
                        </Box>
                      )}
                    </ListItem>
                  ))}
                </List>
              ) : (
                <Paper sx={{ p: 3, textAlign: 'center', backgroundColor: 'grey.50' }}>
                  <DocumentIcon sx={{ fontSize: 48, color: 'grey.400', mb: 2 }} />
                  <Typography variant="h6" color="text.secondary" gutterBottom>
                    No Activity Yet
                  </Typography>
                  <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                    Upload your first contract to get started with AI-powered reviews
                  </Typography>
                  <Button
                    variant="contained"
                    startIcon={<UploadIcon />}
                    onClick={() => navigate('/contract-review')}
                  >
                    Start First Review
                  </Button>
                </Paper>
              )}

              {recentActivity.length > 0 && (
                <Box sx={{ mt: 2, textAlign: 'center' }}>
                  <Button
                    variant="text"
                    onClick={() => navigate('/review-history')}
                  >
                    View All Activity
                  </Button>
                </Box>
              )}
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
}

export default Dashboard; 