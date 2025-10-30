MOD = 998244353

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    K = int(data[1])
    P = list(map(int, data[2:2+N]))
    
    # Compute inversion count inv_P
    inv_P = 0
    fenwick = [0] * (N + 2)
    for i in range(N-1, -1, -1):
        x = P[i]
        cnt = 0
        while x > 0:
            cnt += fenwick[x]
            x -= x & -x
        inv_P += cnt
        inv_P %= MOD
        fenwick[x] += 1
    
    # Precompute factorials and inverses
    fact = [1] * (N + 1)
    for i in range(1, N + 1):
        fact[i] = fact[i-1] * i % MOD
    inv_fact = [1] * (N + 1)
    inv_fact[N] = pow(fact[N], MOD-2, MOD)
    for i in range(N-1, -1, -1):
        inv_fact[i] = inv_fact[i+1] * (i+1) % MOD
    
    inv_2 = pow(2, MOD-2, MOD)
    
    class SegmentTree:
        def __init__(self, data):
            self.n = len(data)
            self.size = 1
            while self.size < self.n:
                self.size <<= 1
            self.tree_count = [0] * (2 * self.size)
            self.tree_sum = [0] * (2 * self.size)
            for i in range(self.n):
                self.tree_count[self.size + i] = 1
                self.tree_sum[self.size + i] = data[i]
            for i in range(self.size - 1, 0, -1):
                self.tree_count[i] = self.tree_count[2*i] + self.tree_count[2*i+1]
                self.tree_sum[i] = self.tree_sum[2*i] + self.tree_sum[2*i+1]
        
        def query_count_sum(self, l, r, x):
            res_count = 0
            res_sum = 0
            l += self.size
            r += self.size
            while l <= r:
                if l % 2 == 1:
                    pos = l
                    while pos <= r:
                        if x > P[pos - self.size]:
                            res_count += self.tree_count[pos]
                            res_sum += self.tree_sum[pos]
                        pos += 1
                    l += 1
                if r % 2 == 0:
                    pos = r
                    while l <= pos:
                        if x > P[pos - self.size]:
                            res_count += self.tree_count[pos]
                            res_sum += self.tree_sum[pos]
                        pos -= 1
                    r -= 1
                l >>= 1
                r >>= 1
            return res_count, res_sum
    
    st = SegmentTree(P)
    
    T = 0
    M = N - K + 1
    if M == 0:
        term3 = 0
    else:
        inv_M = pow(M, MOD-2, MOD)
        term3 = T * inv_M % MOD
    
    for a in range(N):
        L = a + 1
        R = a + K - 1
        if L > R:
            continue
        if L > N-1 or R >= N:
            continue
        x = P[a]
        count, sum_ = st.query_count_sum(L, R, x)
        contribution = ( (a + K) * count - sum_ ) % MOD
        T = (T + contribution) % MOD
    
    inv_P_mod = inv_P % MOD
    term1 = K * (K-1) // 2 % MOD
    term1 = term1 * inv_2 % MOD
    term2 = inv_P_mod
    term3_mod = term3 % MOD
    E = (term1 + term2 - term3_mod) % MOD
    E = (E + MOD) % MOD
    
    print(E)
    
if __name__ == '__main__':
    main()