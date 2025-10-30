n = int(input())
A = list(map(int, input().split()))

def can_form_pyramid(peak, k):
    """Check if we can form a pyramid of size k with peak at position peak"""
    # Peak value must be at least k
    if k > A[peak]:
        return False
    
    # Check if we have enough positions on both sides
    if peak < k - 1 or peak + k - 1 >= n:
        return False
    
    # Check all positions in the pyramid
    for j in range(1, k):
        # Left side: need A[peak-j] >= k-j
        if A[peak - j] < k - j:
            return False
        # Right side: need A[peak+j] >= k-j  
        if A[peak + j] < k - j:
            return False
    
    return True

max_size = 0

# Try each position as the peak
for peak in range(n):
    # Binary search for the maximum valid k
    left, right = 1, min(A[peak], peak + 1, n - peak)
    
    while left <= right:
        mid = (left + right) // 2
        if can_form_pyramid(peak, mid):
            max_size = max(max_size, mid)
            left = mid + 1
        else:
            right = mid - 1

print(max_size)