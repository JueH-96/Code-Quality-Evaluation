def count_permutations(A):
    N = len(A)
    MOD = 998244353
    
    # Convert A to 1-indexed for easier handling
    A = [0] + A
    
    P = [0] * (N + 1)  # 1-indexed permutation
    used = [False] * (N + 1)  # Track used values
    
    def dfs(pos):
        if pos > N:
            return 1
        
        count = 0
        
        # Try all values from 1 to N
        for val in range(1, N + 1):
            if used[val]:  # Skip if value has already been used
                continue
            
            valid = True
            
            # Check the first constraint:
            # For any j with A[pos] < j < pos, P[j] > val
            for j in range(A[pos] + 1, pos):
                if P[j] <= val:
                    valid = False
                    break
            
            # Check the second constraint:
            # If A[pos] > 0, then P[A[pos]] < val
            if A[pos] > 0 and P[A[pos]] >= val:
                valid = False
            
            if valid:
                P[pos] = val
                used[val] = True
                count = (count + dfs(pos + 1)) % MOD
                # Backtrack
                P[pos] = 0
                used[val] = False
        
        return count
    
    return dfs(1)

# Read input
N = int(input())
A = list(map(int, input().split()))

# Compute and print the result
print(count_permutations(A))