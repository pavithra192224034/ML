import numpy as np
from scipy.stats import multivariate_normal

class GaussianMixtureModel:
    def __init__(self, n_components, n_iterations, tol=1e-6):
        self.n_components = n_components
        self.n_iterations = n_iterations
        self.tol = tol
    
    def initialize_parameters(self, X):
        n_samples, n_features = X.shape
        
        self.weights = np.full(self.n_components, 1 / self.n_components)
        random_row = np.random.randint(low=0, high=n_samples, size=self.n_components)
        self.means = X[random_row]
        self.covariances = np.array([np.cov(X, rowvar=False)] * self.n_components)
    
    def e_step(self, X):
        self.resp = np.zeros((X.shape[0], self.n_components))
        
        for k in range(self.n_components):
            self.resp[:, k] = self.weights[k] * multivariate_normal(self.means[k], self.covariances[k]).pdf(X)
        
        self.resp = self.resp / self.resp.sum(axis=1, keepdims=True)
    
    def m_step(self, X):
        resp_sum = self.resp.sum(axis=0)
        
        self.weights = resp_sum / X.shape[0]
        self.means = np.dot(self.resp.T, X) / resp_sum[:, np.newaxis]
        
        for k in range(self.n_components):
            diff = X - self.means[k]
            self.covariances[k] = np.dot(self.resp[:, k] * diff.T, diff) / resp_sum[k]
    
    def compute_log_likelihood(self, X):
        log_likelihood = np.zeros(X.shape[0])
        
        for k in range(self.n_components):
            log_likelihood += self.weights[k] * multivariate_normal(self.means[k], self.covariances[k]).pdf(X)
        
        return np.sum(np.log(log_likelihood))
    
    def fit(self, X):
        self.initialize_parameters(X)
        
        log_likelihood = 0
        for i in range(self.n_iterations):
            self.e_step(X)
            self.m_step(X)
            
            new_log_likelihood = self.compute_log_likelihood(X)
            
            if np.abs(new_log_likelihood - log_likelihood) < self.tol:
                break
            
            log_likelihood = new_log_likelihood
        
        return self
    
    def predict(self, X):
        self.e_step(X)
        return np.argmax(self.resp, axis=1)


if __name__ == "__main__":
   
    np.random.seed(0)
    X = np.vstack([
        np.random.multivariate_normal([0, 0], [[1, 0.75], [0.75, 1]], 400),
        np.random.multivariate_normal([5, 5], [[1, 0.75], [0.75, 1]], 400),
        np.random.multivariate_normal([0, 5], [[1, 0.75], [0.75, 1]], 400)
    ])
    
    
    gmm = GaussianMixtureModel(n_components=3, n_iterations=100)
    gmm.fit(X)
    
    
    labels = gmm.predict(X)
    
   
    import matplotlib.pyplot as plt
    plt.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis')
    plt.show()
