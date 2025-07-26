Q: Is $||\delta_{t+1}|| / ||\delta_t||$ bounded for infinite norm

- A: $||x||_\infty <= ||x||_2 \le ||x||_1 \le sqrt{n} ||x||_2 \le n||x||_\infty$
- A: For $\lambda = 1$, we only needs very few itertions
- $\lambda =0.0001$, the norm ratio is above 7 for 100 iterations (same for $\lambda=0.01$)
- $\lambda=0.5$, the norm ratio is 4.5
- $\lambda=0.9$, the norm ratio is 3
- $\lambda=1$, the norm ratio is below 2,
- $\lambda=1.5$, the norm ratio is above 1.6,
- $\lambda=1.9$, the norm ratio has a spike at 1.2, but reduce to 1.1
- $\lambda=2.$, the norm ratio has a spike at 1.1, but reduce to 1.
- $\lambda=3.$, the norm ratio has a spike at 1.05, but reduce to 1.
- $\lambda=0$, the norm ratio is 1