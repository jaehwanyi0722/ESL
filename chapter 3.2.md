## 3.3 <u> Subset selection</u>

---

### Main objective : Reduce MSE(prediction accuracy)

$$
MSE = (Bias)^2 + Variance
$$

![1](/Users/ijaehwan/Desktop/1.png)

- LSE estimates with all predictors tend to have low bias but high variance(R-squared is improved when introducing new variables)
- Thus we have to consider methods to reduce variance tremedously while allowing little bias

---

### => There are three solutions

#### 1. Selection of Variables

#### 2. Shrinkage Methods: Ridge and Lasso

#### 3. Principal Component Analysis

---

### 1. Selection of variables

##### - Best subset selection

- consider all possible subsets : for each k, find the subset of size k* that gives the smallest RSS

-  infeasible if p is large , since   $\sum_{k=0}^{p}\binom{p}{k}=2^p$

##### - Stepwise Selection

- Forward selection: Begins with the null model and at each subsequent step, adds most significant variables

- Backward elimination: Begins with the full model and at each step removes the least significant variables until none meets the criterion

- Bidirectional selection: Alternates between forward and backward

- Drawbacks: It is unstable and emits high variance

  $\because$ it searches a large space of possible models; performs better in sample than does on new out-of-sample data

##### - Forward-Stagewise Regression

![2](/Users/ijaehwan/Desktop/2.jpg)

- Starts with coefficient vector with zero vector, and set $\epsilon$ >0 (in our case, $\epsilon=1$) 
- At each step, it identifies the variable most correlated with the current residual
- It then computes the simple linear regression coefficient of the model on the chosen variable, and adds to the current coefficient for that variable
- Continued until none of the variables have correlation with the residuals

---

### 2. Shrinkage Methods

##### - Ridge Regression

$$
\hat{\beta}^{Ridge}  = argmin_\beta \sum_{i=1}^{N} (y_i-\beta_0-\sum_{j=1}^{p}\beta_jx_{ij})^2
$$

$$
s.t \sum_{j=1}^{p} \beta_j^2 \le s
$$

<=> 
$$
\hat{\beta}^{Ridge}  = argmin_\beta[\sum_{i=1}^{N}(y_i-\beta_0-\sum_{j=1}^{p}\beta_jx_{ij})^2+\lambda\sum_{j=1}^{p}\beta_j^2]
$$
($\because$ the least squares and sum of sqares of beta coefficients are semi-positive definite => convex => apply KKT condition)

- $\lambda$ is called shrinkage factor ; the greater the value, more shrinkage occurs
- The objective $\beta's$ are equivalent under centered data (refer to lecture note)

Solutions (assume  $\bold{y}\ and\ \bold{X}$ are centered and $X^TX=I$

   $RSS(\lambda)=\ \bold{(y-X\beta)^T(y-X\beta)+\lambda\beta^T\ \beta}$

   $\hat{\beta}^{Ridge}$ $=$ $(X^TX+\lambda I)^{-1}X^Ty$

​              = $\frac{1}{(1+\lambda)} \bold{X^T}\bold{y}$ = $\frac{1}{1+\lambda}\hat{\beta}^{LSE}$

- so if $\lambda$ is small, $LSE \ \approx\ LASSO$ 
- if $\lambda$ is large, $LSE >> LASSO$

- Note that obtaining minimum in Ridge regression is equivalent to obtaining the mean of the posterior distribution in Bayesian persepective

![KakaoTalk_Photo_2021-09-16-14-10-30](/Users/ijaehwan/Desktop/KakaoTalk_Photo_2021-09-16-14-10-30.jpeg)



##### <Reference: SVD of X>

- Suppose $X$ has full rank

- Then by singular value decomposition, $X=UDV^T$, where $U$ and $V$ are orthogonal matrices with dimension $n {\times}p $ and $p {\times} p$ , respectively and $D$ an $p \times p$ diagonal matrix

  - Lemma) For $m \times n$ matrix A with full rank, $n=dim(col(A))=dim(row(A))$

  ![KakaoTalk_Photo_2021-09-16-14-47-10](/Users/ijaehwan/Desktop/KakaoTalk_Photo_2021-09-16-14-47-10.jpeg)

  

  ![KakaoTalk_Photo_2021-09-16-14-47-14](/Users/ijaehwan/Desktop/KakaoTalk_Photo_2021-09-16-14-47-14.jpeg)

  - Lemma) If X is orthogonal matrix: $X^TX=I$ since $X_{(i)} X_{(j)}=0$ if $i \neq j$

  - Lemma) Since $X^TX$ is positive semi-definite, its eigenvalues are nonnegative. Hence the eigen-decomposition of $X$ by taking square root of eigenvalues of $X^TX$ are also nonnegative

    ![KakaoTalk_Photo_2021-09-16-15-14-42](/Users/ijaehwan/Desktop/KakaoTalk_Photo_2021-09-16-15-14-42.jpeg)

- Hence we can say that

  - $ {any\ column \ of \ X} \in C(U)$ and $any \ row \ of \ X \in C(V)$

  - $D$ has $d_1 \geq d_2 \geq...\geq d_p\geq0$ as singular values

- Using the fact that  $X=UDV^T$ and $U\ and\ V$ are orthogonal matrix, we have

​       $X\hat{\beta}^{LSE}$ = $UU^Ty$

​                    = $\sum_{j=1}^{p} u_ju_j^Ty$   

   and

​      $X\hat{\beta}^{Ridge}$ = $UD(D^2+\lambda I)^{-1}DU^T$

​                    = $\sum_{j=1}^{p} \frac{d_j^2}{d_j^2+\lambda}u_ju_j^Ty$

- This can be viewed as projecting $y$ into each orthogonal column spaces, but for ridge regression it is shrunken toward zero
- For given $\lambda$, greater amount of shrinkage is applied with small $d_j's$



##### <Reference: Interpretation of $d_j's$ >

- Consider a centered matrix $X$ and eigen decomposition of $X^T X$ = $VD^2V^{-1}$

- Note that minimizing RSS is equivalent to maximizing variance of projections(refer to [http://www.stat.cmu.edu/~cshalizi/uADA/12/lectures/ch18.pdf]())

- $v_j$: j-th principal component direction

- $z_j$: j-th principal components

  ($z_j = Xv_j=UDV^{T}v_j=d_ju_j$, since $V$ is orthogonal matrix)

  => the first principal component direction has the property that $z_1 = Xv_1$ has the largest variance amongst all normalized linear combinations of columns of X, where $sample\ variance\ of \ z_j=\frac{d_j^2}{N}$

- Hence in order to ==maximize variance==, we should shrink variable most that has the ==minimum variance== => Ridge regression projects $y$ onto each components and shrinks coefficients of low-variance component

![3](/Users/ijaehwan/Desktop/3.png)



##### - Lasso Regression

$$
\hat{\beta}^{Lasso} = argmin_{\beta} \sum_{i=1}^{N}(y_i-\beta_0-\sum_{j=1}^{p}\beta_jx_{ij})^2

\\s.t \sum_{j=1}^{p}\abs{\beta_j}\leq t
$$

<=>
$$
\hat{\beta}^{Lasso} = argmin_{\beta}[\sum_{i=1}^{N}(y_i-\beta_0-\sum_{j=1}^{p}\beta_jx_{ij})^2 + \lambda \sum_{j=1}^{p}\abs{\beta_j}]
$$

- unlike ridge regression, it makes some of the coefficients exactly zero for small $t$

- When $X$ is orthogonal, we have
  $$
  \hat{\beta}^{Lasso} = sign(\hat{\beta}^{LSE})(\abs{\hat{\beta}^{LSE}}-\frac{\lambda}{2})_+
  $$

Which can be interpreted as when $X$ is orthogonal, lasso estimator is equivalent to LSE estimator with soft-thresholding

![KakaoTalk_Photo_2021-09-16-17-43-14](/Users/ijaehwan/Desktop/KakaoTalk_Photo_2021-09-16-17-43-14.jpeg)

![KakaoTalk_Photo_2021-09-16-17-43-20](/Users/ijaehwan/Desktop/KakaoTalk_Photo_2021-09-16-17-43-20.jpeg)

- For computation, Quadratic Programming can be used in two ways:

  - 1. QP with p variables and $2^p$ constraints:

       Let $\delta_i$, $i=1,...,2^p$ be p-tuples of the form $(\pm1,\pm1,...,\pm1)$ . Then $\sum_{j=1}^{p}\abs{\beta_j}\leq t$ is equivalent to $\delta_i^T\beta\leq t$ for all $i$.

    2. QP with 2p variables and (2p+1) constraints:

       Let $\beta_j=\beta_j^+-\beta_j^-$ where $\beta_j^+ \geq 0, \ \beta_j^-\geq0$. And we have the constraints $\beta_j^+ \geq 0, \ \beta_j^-\geq0$ for all $j=1,...,p$ and $\sum\beta_j^+ + \beta_j^- \leq t$.



##### -Comparison of Ridge and Lasso

![4](/Users/ijaehwan/Desktop/4.png)

![5](/Users/ijaehwan/Desktop/5.png)

- Shrinkage factor: $s=t/\sum\abs{\beta_j}$, where $\sum\abs{\beta_j}\leq t$

![6](/Users/ijaehwan/Desktop/6.png)

- Lasso is sparse so tends to provide variable selection

- On the other hand, Ridge makes variables all shrink to zero consistently

  (refer to: [https://stats.stackexchange.com/questions/74542/why-does-the-lasso-provide-variable-selection](https://stats.stackexchange.com/questions/74542/why-does-the-lasso-provide-variable-selection))



##### <Reference: Model degrees of Freedom>

- Any linear estimator $\hat{y}=Sy$ where $S$ is smoother matrix is an $N\times N$ matrix depending on $x_i's$ and $\lambda$ 
- When viewing degress of freedom as rank(S), it can be discontinuous and be problematic
- Thus considering it as trace(S) solves the problem by continuously adjusting the d.f









##### 코드 구현: Forward stagewise regression, ridge, lasso 성능 비교 평가해보기
