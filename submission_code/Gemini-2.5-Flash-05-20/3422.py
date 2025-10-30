class Solution:
    def valueAfterKSeconds(self, n: int, k: int) -> int:
        MOD = 10**9 + 7

        # The problem asks for the value of a[n-1] after k seconds.
        # Let a_t[i] be the value of the i-th element after t seconds.
        # Initial state (t=0): a_0[i] = 1 for all i.
        #
        # The update rule is: a_t[i] = sum(a_{t-1}[0]...a_{t-1}[i])
        #
        # Let's trace the values and observe the pattern:
        # a_0[i] = 1
        # a_1[i] = 1 + 1 + ... + 1 (i+1 times) = i + 1
        # a_2[i] = sum(a_1[0]...a_1[i]) = sum(j+1 for j=0 to i) = sum(val for val=1 to i+1) = (i+1)(i+2)/2
        #
        # These values correspond to binomial coefficients:
        # a_0[i] = 1 = C(i, i)
        # a_1[i] = i + 1 = C(i+1, i)
        # a_2[i] = (i+1)(i+2)/2 = C(i+2, i)
        #
        # In general, it appears that a_t[i] = C(t + i, i).
        # This can be proven by induction using the Hockey-stick identity:
        # sum_{j=0 to k} C(r+j, j) = C(r+k+1, k).
        # If a_{t-1}[j] = C((t-1)+j, j), then
        # a_t[i] = sum_{j=0 to i} C((t-1)+j, j) = C((t-1)+i+1, i) = C(t+i, i).
        #
        # We need to find a_k[n-1]. Using the formula:
        # a_k[n-1] = C(k + (n-1), n-1) = C(n + k - 1, n - 1).

        # Calculate N and K for C(N, K)
        N_val = n + k - 1
        K_val = n - 1

        # We can use C(N, K) = C(N, N-K) to reduce computation if K > N/2.
        # Here, N - K_val = (n + k - 1) - (n - 1) = k.
        # So we want to compute C(n + k - 1, min(n - 1, k)).
        K_val = min(K_val, k)

        # Initialize the result for C(N_val, K_val)
        ans = 1

        # Calculate C(N_val, K_val) = product_{i=0}^{K_val-1} ((N_val - i) / (i + 1))
        # Since we are working modulo a prime, division by (i+1) is multiplication by its modular inverse.
        # Modular inverse of x modulo MOD is x^(MOD-2) by Fermat's Little Theorem.
        for i in range(K_val):
            # Multiply by (N_val - i)
            ans = (ans * (N_val - i)) % MOD
            # Multiply by the modular inverse of (i + 1)
            ans = (ans * pow(i + 1, MOD - 2, MOD)) % MOD
            
        return ans