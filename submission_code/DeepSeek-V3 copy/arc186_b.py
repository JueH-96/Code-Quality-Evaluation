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
    
    # Function to compute combination C(n, k)
    def comb(n, k):
        if k < 0 or k > n:
            return 0
        return fact[n] * inv_fact[k] % MOD * inv_fact[n-k] % MOD
    
    # We need to find the number of valid permutations
    # The conditions can be translated into constraints on the order of elements
    # Specifically, for each i, P_i must be greater than all P_j for A_i < j < i
    # and P_{A_i} < P_i if A_i > 0
    
    # To handle this, we can think of the problem as building the permutation step by step
    # and ensuring that the constraints are satisfied at each step
    
    # We can use a dynamic programming approach where we keep track of the number of ways
    # to arrange the elements up to each index, considering the constraints
    
    # However, a more efficient approach is to recognize that the constraints form a tree-like structure
    # where each element has a parent (A_i) and must be greater than all its siblings
    
    # We can model this as a tree where each node has children that are the elements j where A_j = i
    # and the order of children must be such that the permutation constraints are satisfied
    
    # The number of ways to arrange the children of a node is the factorial of the number of children
    # multiplied by the number of ways to arrange each subtree
    
    # So, the total number of valid permutations is the product of the factorials of the number of children
    # for each node, multiplied by the number of ways to arrange the subtrees
    
    # To implement this, we can build a tree where each node has a list of children
    # and then perform a post-order traversal to compute the number of ways
    
    # First, build the tree
    children = defaultdict(list)
    for i in range(N):
        children[A[i]].append(i+1)  # since A_i is 0-based and P is 1-based
    
    # Now, perform a post-order traversal to compute the number of ways
    # We will use a recursive function with memoization
    
    from functools import lru_cache
    
    @lru_cache(maxsize=None)
    def dfs(u):
        if not children[u]:
            return 1
        total = 1
        cnt = 0
        for v in children[u]:
            cnt += 1
            total = total * dfs(v) % MOD
        total = total * fact[cnt] % MOD
        return total
    
    result = dfs(0)
    print(result)

if __name__ == "__main__":
    main()