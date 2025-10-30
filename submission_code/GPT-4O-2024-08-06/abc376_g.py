# YOUR CODE HERE
import sys
from collections import defaultdict
from functools import lru_cache

MOD = 998244353

def mod_inv(x, mod):
    """ Compute the modular inverse of x under modulo mod """
    return pow(x, mod - 2, mod)

def solve():
    input = sys.stdin.read
    data = input().split()
    
    index = 0
    T = int(data[index])
    index += 1
    results = []
    
    for _ in range(T):
        N = int(data[index])
        index += 1
        p = list(map(int, data[index:index + N]))
        index += N
        a = list(map(int, data[index:index + N]))
        index += N
        
        # Build the tree
        children = defaultdict(list)
        for i in range(N):
            children[p[i]].append(i + 1)
        
        # Calculate the sum of a_i
        sum_a = sum(a)
        
        # Calculate expected operations using DFS
        @lru_cache(None)
        def dfs(node):
            if node not in children:
                return 0
            total = 0
            for child in children[node]:
                total += (a[child - 1] * (1 + dfs(child))) % MOD
            return total * mod_inv(sum_a, MOD) % MOD
        
        # Calculate the expected number of operations
        expected_operations = dfs(0)
        results.append(expected_operations)
    
    for result in results:
        print(result)