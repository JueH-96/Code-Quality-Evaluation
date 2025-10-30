import sys

def solve():
    """
    Solves the Expected Inversion Number problem.
    """
    MOD = 998244353

    def power(a, b):
        """
        Calculates (a^b) % MOD using modular exponentiation.
        """
        res = 1
        a %= MOD
        while b > 0:
            if b % 2 == 1:
                res = (res * a) % MOD
            a = (a * a) % MOD
            b //= 2
        return res

    def inv(n):
        """
        Calculates the modular inverse of n modulo MOD.
        """
        return power(n, MOD - 2)

    class FenwickTree:
        """
        Fenwick Tree (Binary Indexed Tree) for sum queries.
        Uses 1-based indexing for tree and input indices.
        """
        def __init__(self, size):
            self.tree = [0] * (size + 1)
            self.size = size

        def add(self, i, x):
            while i <= self.size:
                self.tree[i] += x
                i += i & -i

        def query(self, i):
            """
            Calculates sum of elements in the range [1, i].
            """
            s = 0
            while i > 0:
                s += self.tree[i]
                i -= i & -i
            return s

    # Read input
    try:
        input = sys.stdin.readline
        N, K = map(int, input().split())
        P = list(map(int, input().split()))
    except (IOError, ValueError):
        # Handle potential empty input for local testing
        return

    # 1. Calculate initial number of inversions (I_0)
    bit_I0 = FenwickTree(N)
    I_0 = 0
    for i in range(N - 1, -1, -1):
        I_0 = (I_0 + bit_I0.query(P[i] - 1)) % MOD
        bit_I0.add(P[i], 1)

    if K == 1:
        print(I_0)
        return

    # 2. Expected number of inversions within any shuffled window
    inv4 = inv(4)
    term2 = (K * (K - 1)) % MOD * inv4 % MOD

    # 3. Calculate Sum_Z = sum_{s=0}^{N-K} Z_s, where Z_s is the inversion
    # count in the window P[s...s+K-1], using a sliding window with a BIT.
    
    bit_Z = FenwickTree(N)
    current_Z = 0
    
    # Initialize for the first window P[0...K-1]
    for i in range(K - 1, -1, -1):
        p_i = P[i]
        current_Z = (current_Z + bit_Z.query(p_i - 1))
        bit_Z.add(p_i, 1)
    current_Z %= MOD
    Sum_Z = current_Z
    
    # Slide the window from s=0 to N-K-1
    for s in range(N - K):
        # Window moves from [s, s+K-1] to [s+1, s+K]
        p_s = P[s]
        p_sk = P[s+K]

        # Update current_Z based on elements entering and leaving the window.
        # Remove contribution of p_s: inversions (p_s, p_j) for j in [s+1, s+K-1]
        bit_Z.add(p_s, -1)
        loss = bit_Z.query(p_s - 1)
        
        # Add contribution of p_sk: inversions (p_i, p_sk) for i in [s+1, s+K-1]
        gain = (K - 1) - bit_Z.query(p_sk)
        
        current_Z = (current_Z - loss + gain + MOD) % MOD
        Sum_Z = (Sum_Z + current_Z) % MOD
        
        bit_Z.add(p_sk, 1)
        
    # 4. Final Calculation
    # E = I_0 + K*(K-1)/4 - (Sum_Z / (N-K+1))
    inv_denom = inv(N - K + 1)
    term3 = (Sum_Z * inv_denom) % MOD
    
    result = (I_0 + term2 - term3 + MOD) % MOD
    
    print(result)

solve()