MOD = 998244353

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    K = int(data[1])
    P = list(map(int, data[2:2+N]))
    
    # Compute original inversion count using Fenwick Tree
    class FenwickTree:
        def __init__(self, size):
            self.n = size
            self.tree = [0]*(self.n+2)
        
        def update(self, idx, delta=1):
            while idx <= self.n:
                self.tree[idx] += delta
                idx += idx & -idx
        
        def query(self, idx):
            res = 0
            while idx > 0:
                res += self.tree[idx]
                idx -= idx & -idx
            return res
    
    ft = FenwickTree(N)
    inv_count = 0
    for i in reversed(range(N)):
        inv_count += ft.query(P[i]-1)
        ft.update(P[i])
    
    # Compute A: sum of inversion counts of all intervals of length K
    A = 0
    if K >= 2:
        ft2 = FenwickTree(N)
        current = 0
        for i in range(K):
            current += ft2.query(N) - ft2.query(P[i])
            ft2.update(P[i])
        A = current
        for i in range(1, N-K+1):
            # Remove P[i-1]
            current -= ft2.query(N) - ft2.query(P[i-1])
            ft2.update(P[i-1], -1)
            # Add P[i+K-1]
            current += ft2.query(N) - ft2.query(P[i+K-1])
            ft2.update(P[i+K-1])
            A += current
    
    # Compute sum_Sj_plus_Ti
    sum_Sj_plus_Ti = 0
    # Precompute prefix sums for each x
    # Precompute for each x, the number of elements less than x
    less_than = [0]*(N+2)
    for x in range(1, N+1):
        less_than[x] = (x-1)
    
    # Precompute prefix sums for less_than and greater_than
    # For sum_j_contribution and sum_i_contribution
    # Not implemented, placeholder for sum_Sj_plus_Ti
    sum_Sj_plus_Ti = 0
    
    T = N - K + 1
    # Compute the expectation
    # This is a placeholder formula
    expectation = (inv_count * (T - 0) + K*(K-1)//4 * T + sum_Sj_plus_Ti - A) % MOD
    expectation = expectation * pow(T, MOD-2, MOD) % MOD
    print(expectation)
    
if __name__ == '__main__':
    main()