import React from 'react';

export const Button = ({ 
  children, 
  variant = 'default', 
  size = 'default', 
  className = '', 
  disabled = false,
  type = 'button',
  onClick,
  ...props 
}) => {
  const baseClasses = 'inline-flex items-center justify-center rounded-md font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:opacity-50 disabled:pointer-events-none';
  
  const variants = {
    default: 'bg-blue-600 text-white hover:bg-blue-700',
    outline: 'border border-gray-300 bg-white text-gray-700 hover:bg-gray-50',
    secondary: 'bg-gray-100 text-gray-900 hover:bg-gray-200',
    destructive: 'bg-red-600 text-white hover:bg-red-700'
  };

  const sizes = {
    default: 'h-10 py-2 px-4',
    sm: 'h-9 px-3 text-sm',
    lg: 'h-11 px-8',
    icon: 'h-10 w-10'
  };

  const variantClasses = variants[variant] || variants.default;
  const sizeClasses = sizes[size] || sizes.default;

  return (
    <button
      type={type}
      className={`${baseClasses} ${variantClasses} ${sizeClasses} ${className}`}
      disabled={disabled}
      onClick={onClick}
      {...props}
    >
      {children}
    </button>
  );
}; 