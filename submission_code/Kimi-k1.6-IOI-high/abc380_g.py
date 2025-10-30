MOD = 998244353

class FenwickTree:
    def __init__(self, size):
        self.n = size
        self.tree = [0] * (self.n + 2)  # 1-based indexing

    def update(self, idx, delta):
        # idx is 1-based
        while idx <= self.n:
            self.tree[idx] += delta
            idx += idx & -idx

    def query(self, idx):
        # sum from 1 to idx (inclusive)
        res = 0
        while idx > 0:
            res += self.tree[idx]
            idx -= idx & -idx
        return res

def compute_original_inv(P, MOD):
    n = len(P)
    ft = FenwickTree(n)
    original_inv = 0
    for i in reversed(range(n)):
        val = P[i]
        original_inv += ft.query(val - 1)
        original_inv %= MOD
        ft.update(val, 1)
    return original_inv % MOD

def compute_sum_inv_internal(P, K, MOD):
    N = len(P)
    if K == 0 or K > N:
        return 0
    ft = FenwickTree(N)
    current_inversion = 0
    # Initialize the first window [0, K-1]
    for i in range(K-1, -1, -1):
        val = P[i]
        current_inversion += ft.query(val - 1)
        current_inversion %= MOD
        ft.update(val, 1)
    sum_inv_internal = current_inversion % MOD
    # Slide the window N-K times
    for i in range(0, N - K):
        # Remove P[i]
        val = P[i]
        ft.update(val, -1)
        # Compute count_less
        count_less = ft.query(val - 1)
        current_inversion = (current_inversion - count_less) % MOD
        # Add new_val P[i+K]
        new_val = P[i + K]
        count_greater = (ft.query(N) - ft.query(new_val)) % MOD
        current_inversion = (current_inversion + count_greater) % MOD
        ft.update(new_val, 1)
        sum_inv_internal = (sum_inv_internal + current_inversion) % MOD
    return sum_inv_internal % MOD

def main():
    import sys
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    K = int(input[ptr])
    ptr += 1
    P = list(map(int, input[ptr:ptr + N]))
    ptr += N
    
    if K == 0:
        print(0)
        return
    
    original_inv = compute_original_inv(P, MOD)
    sum_inv_internal = compute_sum_inv_internal(P, K, MOD)
    
    D = N - K + 1
    invD = pow(D, MOD - 2, MOD)
    inv4 = pow(4, MOD - 2, MOD)
    
    term1 = (original_inv % MOD) * D % MOD
    term2 = K % MOD
    term2 = term2 * ((K - 1) % MOD) % MOD
    term2 = term2 * D % MOD
    term2 = term2 * inv4 % MOD
    term3 = sum_inv_internal % MOD
    numerator = (term1 + term2 - term3) % MOD
    ans = numerator * invD % MOD
    print(ans)

if __name__ == '__main__':
    main()