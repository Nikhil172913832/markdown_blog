Bayesian Optimization is a **global optimization technique** that finds the best hyperparameters **by modeling the objective function probabilistically** instead of blindly searching. The math involves two key components:

1. **Surrogate Function (Gaussian Process - GP)**
    
2. **Acquisition Function (Expected Improvement, Upper Confidence Bound, etc.)**
    

---

## **1. Surrogate Function: Gaussian Process (GP) Regression**

Since evaluating hyperparameters is expensive (e.g., training deep networks takes hours), we approximate the objective function f(x)f(x)f(x) using a surrogate function.

A **Gaussian Process (GP)** is a **non-parametric** model that estimates f(x)f(x)f(x) with **a mean and uncertainty (variance) at every point**.

### **Gaussian Process Definition**

A GP defines a **distribution over functions**:

f(x)∼GP(m(x),k(x,x′))f(x) \sim \mathcal{GP}(m(x), k(x, x'))f(x)∼GP(m(x),k(x,x′))

Where:

- m(x)m(x)m(x) is the **mean function** (usually assumed to be 0).
    
- k(x,x′)k(x, x')k(x,x′) is the **covariance/kernel function**, which measures similarity between points xxx and x′x'x′.
    

---

### **Gaussian Process Prior & Posterior**

Initially, before evaluating any points, we assume:

f(x)∼N(0,K)f(x) \sim \mathcal{N}(0, K)f(x)∼N(0,K)

where KKK is the covariance matrix using a kernel function like **Radial Basis Function (RBF)**:

k(x,x′)=exp⁡(−∣∣x−x′∣∣22l2)k(x, x') = \exp\left(-\frac{||x - x'||^2}{2l^2}\right)k(x,x′)=exp(−2l2∣∣x−x′∣∣2​)

After observing some data points (X,y)(X, y)(X,y), we **update our belief** about f(x)f(x)f(x) using **Bayes' Theorem**:

p(f(x)∣X,y)∼N(μ(x),σ2(x))p(f(x) | X, y) \sim \mathcal{N}(\mu(x), \sigma^2(x))p(f(x)∣X,y)∼N(μ(x),σ2(x))

where:

- **Mean μ(x)\mu(x)μ(x)**: Our best estimate of f(x)f(x)f(x).
    
- **Variance σ2(x)\sigma^2(x)σ2(x)**: Our uncertainty at each point.
    

The mean and variance of the Gaussian Process posterior are computed as:

μ(x)=KxXKXX−1y\mu(x) = K_{xX} K_{XX}^{-1} yμ(x)=KxX​KXX−1​y σ2(x)=Kxx−KxXKXX−1KXx\sigma^2(x) = K_{xx} - K_{xX} K_{XX}^{-1} K_{Xx}σ2(x)=Kxx​−KxX​KXX−1​KXx​

where:

- KXXK_{XX}KXX​ is the covariance matrix of observed data.
    
- KxXK_{xX}KxX​ is the covariance between new and observed points.
    
- KxxK_{xx}Kxx​ is the self-covariance of new points.
    

This allows us to **predict the function without explicitly defining it!** 🚀

---

## **2. Acquisition Function: Selecting the Next Point**

Now that we have a Gaussian Process model for f(x)f(x)f(x), we need to decide **where to sample next**. This is done using an **acquisition function** that balances **exploration vs exploitation**.

### **Common Acquisition Functions**

1. **Expected Improvement (EI)**
    
    - Picks points that **maximize expected improvement** over the current best result.
        
    - Formula:
        
        EI(x)=E[max⁡(0,f(x)−f∗)]EI(x) = \mathbb{E}[\max(0, f(x) - f^*)]EI(x)=E[max(0,f(x)−f∗)]
        
        where f∗f^*f∗ is the best observed function value so far.
        
    - This encourages selecting points with high predicted values and high uncertainty.
        
2. **Upper Confidence Bound (UCB)**
    
    - Selects points that maximize:
        
        UCB(x)=μ(x)+κσ(x)UCB(x) = \mu(x) + \kappa \sigma(x)UCB(x)=μ(x)+κσ(x)
    - **κ\kappaκ controls exploration-exploitation** trade-off.
        
        - **Large κ\kappaκ** → More exploration.
            
        - **Small κ\kappaκ** → More exploitation.
            
3. **Probability of Improvement (PI)**
    
    - Picks points where probability of improvement is highest:
        
        PI(x)=P(f(x)>f∗)PI(x) = P(f(x) > f^*)PI(x)=P(f(x)>f∗)
    - This strategy is simple but less effective than EI.
        

---

## **3. Full Bayesian Optimization Algorithm**

Now we put everything together:

1. **Initialize** with a few randomly selected hyperparameters.
    
2. **Fit a Gaussian Process** to model the objective function f(x)f(x)f(x).
    
3. **Use an acquisition function** (EI, UCB, PI) to choose the next best hyperparameter.
    
4. **Evaluate the objective function** at the chosen hyperparameter.
    
5. **Update the Gaussian Process model** with the new observation.
    
6. **Repeat until convergence** (or budget runs out).
    

---

## **4. Why Bayesian Optimization is Efficient**

✅ **Learns from past evaluations** → Doesn't waste time on bad regions.  
✅ **Uncertainty-aware** → Balances exploration and exploitation.  
✅ **Works well with expensive function evaluations** (Deep Learning, Hyperparameter tuning).

🔴 **Downside?**

- **Computational cost** increases as the number of evaluations grows (because GP inversion scales as O(n3)O(n^3)O(n3)).
    
- **Not ideal for very high-dimensional spaces**, but TPE (Tree-structured Parzen Estimators) solve this.