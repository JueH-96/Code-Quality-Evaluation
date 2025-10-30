import sys
from collections import deque

MOD = 998244353

class FenwickTree:
    def __init__(self, size):
        self.size = size
        self.tree = [0] * (self.size + 2)
    
    def update(self, index, delta):
        while index <= self.size:
            self.tree[index] += delta
            index += index & -index
    
    def query(self, index):
        res = 0
        while index > 0:
            res += self.tree[index]
            index -= index & -index
        return res

def main():
    N, K = map(int, sys.stdin.readline().split())
    P = list(map(int, sys.stdin.readline().split()))
    if N == 0 or K == 0:
        print(0)
        return
    
    max_val = N
    
    # Compute initial_inv
    initial_inv = 0
    ft_initial = FenwickTree(max_val)
    for x in reversed(P):
        initial_inv += ft_initial.query(x - 1)
        ft_initial.update(x, 1)
    initial_inv %= MOD
    
    # Compute sum_w_inv
    if K > N:
        K = N
    sum_w_inv = 0
    current_inv = 0
    ft_window = FenwickTree(max_val)
    window = deque()
    
    for i in range(K):
        x = P[i]
        cnt = ft_window.query(max_val) - ft_window.query(x)
        current_inv += cnt
        ft_window.update(x, 1)
        window.append(x)
    sum_w_inv += current_inv
    
    for i in range(K, N):
        x = P[i]
        y = window.popleft()
        cnt = ft_window.query(y - 1)
        current_inv -= cnt
        ft_window.update(y, -1)
        cnt = ft_window.query(max_val) - ft_window.query(x)
        current_inv += cnt
        ft_window.update(x, 1)
        window.append(x)
        sum_w_inv += current_inv
    
    sum_w_inv %= MOD
    
    T = N - K + 1
    if T == 0:
        E = initial_inv % MOD
        print(E)
        return
    
    inv_2 = pow(2, MOD-2, MOD)
    inv_T = pow(T, MOD-2, MOD)
    
    term1 = (K * (K-1) // 2) % MOD
    term1 = (term1 * inv_2) % MOD
    
    term2 = (sum_w_inv * inv_T) % MOD
    
    E = (initial_inv + term1 - term2) % MOD
    E = (E + MOD) % MOD  # Ensure non-negative
    
    print(E)

if __name__ == "__main__":
    main()