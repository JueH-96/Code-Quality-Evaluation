def get_lis_length(arr):
    n = len(arr)
    dp = [1] * n
    for i in range(n):
        for j in range(i):
            if arr[i] > arr[j]:
                dp[i] = max(dp[i], dp[j] + 1)
    return max(dp) if dp else 0

def get_lds_length(arr):
    n = len(arr)
    dp = [1] * n
    for i in range(n):
        for j in range(i):
            if arr[i] < arr[j]:
                dp[i] = max(dp[i], dp[j] + 1)
    return max(dp) if dp else 0

def check_permutation(perm, A, B):
    n = len(perm)
    if get_lis_length(perm) != A or get_lds_length(perm) != B:
        return False
    
    # Check if there exists a value that can be appended
    found = False
    for i in range(1, n+2):
        # Try inserting i + 0.5
        test_perm = list(perm) + [i + 0.5]
        if (get_lis_length(test_perm) == A and 
            get_lds_length(test_perm) == B):
            found = True
            break
    return found

def generate_permutations(n):
    if n == 1:
        return [[1]]
    perms = []
    for perm in generate_permutations(n-1):
        for i in range(n):
            new_perm = perm[:i] + [n] + perm[i:]
            perms.append(new_perm)
    return perms

def solve(A, B, M):
    if A * B > 120:  # According to constraints
        return 0
        
    n = A * B - 1
    count = 0
    
    for perm in generate_permutations(n):
        if check_permutation(perm, A, B):
            count = (count + 1) % M
            
    return count

# Read input
A, B, M = map(int, input().split())

# Solve and output
result = solve(A, B, M)
print(result)