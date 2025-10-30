def max_pyramid_size(A):
    n = len(A)
    max_size = 0
    
    for peak in range(n):
        values = [A[j] + abs(peak - j) for j in range(n)]
        
        for L in range(peak + 1):
            for R in range(peak, n):
                min_value = min(values[L:R+1])
                pyramid_size = min(min_value, (R - L + 2) // 2)
                max_size = max(max_size, pyramid_size)
    
    return max_size

# Read input
n = int(input())
A = list(map(int, input().split()))

# Compute and print result
print(max_pyramid_size(A))