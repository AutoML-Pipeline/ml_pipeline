/**
 * API Test JavaScript
 * Handles API testing functionality in the documentation
 */

document.addEventListener('DOMContentLoaded', function() {
    const apiTestForm = document.getElementById('api-test-form');
    const endpointSelect = document.getElementById('endpoint-select');
    const requestBodyContainer = document.getElementById('request-body-container');
    const fileUploadContainer = document.getElementById('file-upload-container');
    const requestBodyTextarea = document.getElementById('request-body');
    const fileUploadInput = document.getElementById('file-upload');
    const apiResponseContainer = document.getElementById('api-response-container');
    const responseStatus = document.getElementById('response-status');
    const responseTime = document.getElementById('response-time');
    const responseBody = document.getElementById('response-body');

    // Sample request bodies for different endpoints
    const sampleRequests = {
        'POST:/data_ingestion': null, // File upload endpoint - no JSON body
        'POST:/data_preprocessing': {
            "dataset_id": "12345",
            "operations": [
                {"type": "remove_nulls", "columns": ["col1", "col2"]},
                {"type": "fillna", "columns": ["col3"], "value": 0},
                {"type": "normalize", "columns": ["col4", "col5"], "method": "minmax"}
            ],
            "output_name": "preprocessed_dataset"
        },
        'POST:/feature_engineering': {
            "dataset_id": "12345",
            "operations": [
                {"type": "polynomial_features", "columns": ["col1", "col2"], "degree": 2},
                {"type": "binning", "column": "col3", "bins": 5}
            ],
            "output_name": "featured_dataset"
        },
        'POST:/model_selection/recommend': {
            "dataset_id": "12345",
            "target_column": "target",
            "problem_type": "classification",
            "evaluation_metric": "accuracy"
        },
        'POST:/model_selection/compare': {
            "dataset_id": "12345",
            "target_column": "target",
            "problem_type": "classification",
            "models": [
                {"name": "random_forest_classifier", "hyperparameters": {"n_estimators": 100}},
                {"name": "logistic_regression", "hyperparameters": {"C": 1.0}}
            ],
            "evaluation_metrics": ["accuracy", "f1"],
            "cv_folds": 5
        },
        'POST:/model_training_evaluation/train': {
            "dataset_id": "12345",
            "target_column": "target",
            "model_name": "random_forest_classifier",
            "hyperparameters": {
                "n_estimators": 100,
                "max_depth": 10
            },
            "test_size": 0.2,
            "random_state": 42,
            "model_description": "Random Forest model for classification"
        },
        'POST:/visualization/data': {
            "dataset_id": "12345",
            "visualization_type": "histogram",
            "columns": ["numerical_column"],
            "title": "Distribution of Values",
            "options": {
                "bins": 20,
                "figsize": [10, 6]
            }
        },
        'POST:/deployment/deploy': {
            "model_id": "12345",
            "deployment_name": "my-model-endpoint",
            "description": "Deployed model for predictions",
            "version": "1.0"
        },
        'POST:/deployment/{deployment_id}/predict': {
            "data": [
                {"feature1": 1.0, "feature2": 2.0, "feature3": 3.0},
                {"feature1": 4.0, "feature2": 5.0, "feature3": 6.0}
            ]
        }
    };

    // Handle endpoint selection change
    endpointSelect.addEventListener('change', function() {
        const selectedEndpoint = this.value;
        
        if (!selectedEndpoint) {
            requestBodyContainer.style.display = 'none';
            fileUploadContainer.style.display = 'none';
            return;
        }
        
        // Show/hide appropriate input methods based on endpoint
        const [method, path] = selectedEndpoint.split(':');
        
        if (method === 'GET') {
            requestBodyContainer.style.display = 'none';
            fileUploadContainer.style.display = 'none';
        } else if (path === '/data_ingestion' && method === 'POST') {
            // File upload endpoint
            requestBodyContainer.style.display = 'none';
            fileUploadContainer.style.display = 'block';
        } else {
            // JSON body endpoint
            requestBodyContainer.style.display = 'block';
            fileUploadContainer.style.display = 'none';
            
            // Populate with sample request if available
            if (sampleRequests[selectedEndpoint]) {
                requestBodyTextarea.value = JSON.stringify(sampleRequests[selectedEndpoint], null, 4);
            } else {
                requestBodyTextarea.value = '{\n    \n}';
            }
        }
    });

    // Handle form submission
    apiTestForm.addEventListener('submit', function(e) {
        e.preventDefault();
        
        const selectedEndpoint = endpointSelect.value;
        if (!selectedEndpoint) {
            showError('Please select an endpoint');
            return;
        }
        
        const [method, path] = selectedEndpoint.split(':');
        sendApiRequest(method, path);
    });

    // Function to send API request
    function sendApiRequest(method, path) {
        const startTime = new Date().getTime();
        let url = path;
        
        // Replace any path parameters with a placeholder value for demonstration
        if (url.includes('{')) {
            url = url.replace(/{([^}]+)}/g, '12345');
        }
        
        const options = {
            method: method,
            headers: {}
        };

        // Add request body or form data
        if (method !== 'GET') {
            if (path === '/data_ingestion' && method === 'POST') {
                // Handle file upload with FormData
                if (fileUploadInput.files.length === 0) {
                    showError('Please select a file to upload');
                    return;
                }
                
                const formData = new FormData();
                formData.append('file', fileUploadInput.files[0]);
                formData.append('dataset_name', 'Sample Dataset');
                formData.append('description', 'Uploaded from API Tester');
                
                options.body = formData;
                // Don't set Content-Type header for FormData - browser will set it automatically with boundary
            } else {
                // Handle JSON body
                try {
                    const requestBody = JSON.parse(requestBodyTextarea.value);
                    options.body = JSON.stringify(requestBody);
                    options.headers['Content-Type'] = 'application/json';
                } catch (e) {
                    showError('Invalid JSON in request body');
                    return;
                }
            }
        }

        // Show loading state
        responseStatus.textContent = 'Loading...';
        responseStatus.className = 'badge bg-secondary me-2';
        responseTime.textContent = '';
        responseBody.textContent = '';
        apiResponseContainer.style.display = 'block';

        // Send the request
        fetch(url, options)
            .then(response => {
                const endTime = new Date().getTime();
                const duration = endTime - startTime;
                
                responseTime.textContent = `${duration}ms`;
                responseStatus.textContent = response.status + ' ' + response.statusText;
                
                if (response.ok) {
                    responseStatus.className = 'badge bg-success me-2';
                } else {
                    responseStatus.className = 'badge bg-danger me-2';
                }
                
                return response.json();
            })
            .then(data => {
                responseBody.textContent = JSON.stringify(data, null, 2);
            })
            .catch(error => {
                responseStatus.textContent = 'Error';
                responseStatus.className = 'badge bg-danger me-2';
                responseBody.textContent = 'Error: ' + error.message;
            });
    }

    // Function to show error message
    function showError(message) {
        responseStatus.textContent = 'Error';
        responseStatus.className = 'badge bg-danger me-2';
        responseTime.textContent = '';
        responseBody.textContent = message;
        apiResponseContainer.style.display = 'block';
    }
});
