import React, { useState } from 'react';

export const Select = ({ 
  children, 
  value, 
  onValueChange, 
  defaultValue,
  ...props 
}) => {
  const [isOpen, setIsOpen] = useState(false);
  const [selectedValue, setSelectedValue] = useState(value || defaultValue || '');

  const handleValueChange = (newValue) => {
    setSelectedValue(newValue);
    if (onValueChange) {
      onValueChange(newValue);
    }
    setIsOpen(false);
  };

  return (
    <div className="relative">
      {React.Children.map(children, (child) => {
        if (child.type === SelectTrigger) {
          return React.cloneElement(child, { 
            onClick: () => setIsOpen(!isOpen),
            value: selectedValue
          });
        }
        if (child.type === SelectContent) {
          return React.cloneElement(child, { 
            isOpen,
            onValueChange: handleValueChange,
            onClose: () => setIsOpen(false)
          });
        }
        return child;
      })}
    </div>
  );
};

export const SelectTrigger = ({ 
  children, 
  className = '',
  onClick,
  value,
  ...props 
}) => {
  return (
    <button
      type="button"
      className={`flex h-10 w-full items-center justify-between rounded-md border border-gray-300 bg-white px-3 py-2 text-sm placeholder:text-gray-500 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50 ${className}`}
      onClick={onClick}
      {...props}
    >
      {children}
      <svg
        className="h-4 w-4 opacity-50"
        fill="none"
        stroke="currentColor"
        viewBox="0 0 24 24"
      >
        <polyline points="6,9 12,15 18,9"></polyline>
      </svg>
    </button>
  );
};

export const SelectValue = ({ 
  placeholder = 'Select...', 
  value,
  children 
}) => {
  return (
    <span className={value ? '' : 'text-gray-500'}>
      {children || value || placeholder}
    </span>
  );
};

export const SelectContent = ({ 
  children, 
  className = '',
  isOpen,
  onValueChange,
  onClose
}) => {
  if (!isOpen) return null;

  return (
    <>
      <div 
        className="fixed inset-0 z-40" 
        onClick={onClose}
      />
      <div className={`absolute z-50 w-full mt-1 bg-white border border-gray-300 rounded-md shadow-lg max-h-60 overflow-auto ${className}`}>
        {React.Children.map(children, (child) => {
          if (child.type === SelectItem) {
            return React.cloneElement(child, { 
              onValueChange,
              onClose
            });
          }
          return child;
        })}
      </div>
    </>
  );
};

export const SelectItem = ({ 
  children, 
  value, 
  className = '',
  onValueChange,
  onClose,
  ...props 
}) => {
  const handleClick = () => {
    if (onValueChange) {
      onValueChange(value);
    }
    if (onClose) {
      onClose();
    }
  };

  return (
    <div
      className={`px-3 py-2 text-sm cursor-pointer hover:bg-gray-100 ${className}`}
      onClick={handleClick}
      {...props}
    >
      {children}
    </div>
  );
}; 