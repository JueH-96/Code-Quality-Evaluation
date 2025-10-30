# YOUR CODE HERE
import sys
from collections import defaultdict

MOD = 998244353

def main():
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    A = list(map(int, data[1:N+1]))
    
    # Initialize the factorial and inverse factorial arrays
    fact = [1] * (N + 1)
    for i in range(1, N + 1):
        fact[i] = fact[i-1] * i % MOD
    
    inv_fact = [1] * (N + 1)
    inv_fact[N] = pow(fact[N], MOD-2, MOD)
    for i in range(N-1, -1, -1):
        inv_fact[i] = inv_fact[i+1] * (i+1) % MOD
    
    # Build the tree
    children = defaultdict(list)
    for i in range(N):
        children[A[i]].append(i+1)
    
    # DP function to compute the number of ways
    def dfs(u):
        res = 1
        cnt = 0
        for v in children[u]:
            cnt += 1
            res = res * dfs(v) % MOD
        res = res * fact[cnt] % MOD
        return res
    
    total = dfs(0)
    print(total)

if __name__ == "__main__":
    main()