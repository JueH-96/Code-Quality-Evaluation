# YOUR CODE HERE
import sys
import sys
import sys

def solve():
    import sys
    import sys
    sys.setrecursionlimit(1 << 25)
    MOD = 998244353

    class BIT:
        def __init__(self, size):
            self.N = size
            self.tree = [0] * (self.N + 2)

        def update(self, idx, delta):
            while idx <= self.N:
                self.tree[idx] += delta
                idx += idx & -idx

        def query(self, idx):
            res = 0
            while idx > 0:
                res += self.tree[idx]
                idx -= idx & -idx
            return res

    def inv(x):
        return pow(x, MOD - 2, MOD)

    input = sys.stdin.read().split()
    N = int(input[0])
    K = int(input[1])
    P = list(map(int, input[2:2+N]))

    # Compute I_total
    bit_total = BIT(N)
    I_total = 0
    for num in P:
        I_total += bit_total.query(N) - bit_total.query(num)
        I_total %= MOD
        bit_total.update(num, 1)

    # Compute sum_Iin
    M = N - K +1
    if M <=0:
        M =1
    sum_Iin = 0
    bit_window = BIT(N)
    # Initialize window [2, 3, ..., K] if K >=2
    for pos in range(1, min(K, N)):
        bit_window.update(P[pos], 1)
    for a in range(N - M +1):
        a_idx = a
        P_a = P[a_idx]
        count_less = bit_window.query(P_a -1)
        if a_idx +1 <= K -1 and a_idx < M:
            C = a_idx +1
            if C > K -1:
                C = K -1
        elif a_idx < M:
            C = K -1
        else:
            C =0
        if a_idx +1 <= K -1 and a_idx < M:
            C = a_idx +1
        elif a_idx >= K -1 and a_idx < M:
            C = K -1
        else:
            C =0
        sum_Iin = (sum_Iin + (C * count_less) % MOD) % MOD
        # Move the window
        if a_idx + K < N:
            bit_window.update(P[a_idx + K],1)
        if a_idx +1 < N:
            bit_window.update(P[a_idx +1], -1)
    # Correct the above sliding window implementation
    # Re-implement sum_Iin correctly
    sum_Iin =0
    bit_window = BIT(N)
    # Initialize window [2, 3, ..., K] if K >=2
    for pos in range(1, min(K, N)):
        bit_window.update(P[pos], 1)
    for a in range(0, M):
        P_a = P[a]
        count_less = bit_window.query(P_a -1)
        if a < K-1:
            C = a +1
        else:
            C = K -1
        sum_Iin = (sum_Iin + (C * count_less) % MOD) % MOD
        # Update the window
        if a + K < N:
            bit_window.update(P[a + K],1)
        if a +1 < N:
            bit_window.update(P[a +1], -1)
    # Now compute E[I} = I_total - sum_Iin / M + K*(K-1)/4
    invM = inv(M)
    inv4 = inv(4)
    term1 = I_total
    term2 = (sum_Iin * invM) % MOD
    term3 = (K * (K-1)) % MOD
    term3 = (term3 * inv4) % MOD
    E = (term1 - term2 + term3) % MOD
    print(E)