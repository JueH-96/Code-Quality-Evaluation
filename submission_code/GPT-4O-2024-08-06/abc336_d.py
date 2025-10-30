def can_form_pyramid(A, k):
    n = len(A)
    required_length = 2 * k - 1
    if required_length > n:
        return False
    
    # Check if we can form the sequence 1, 2, ..., k, k-1, ..., 2, 1
    left = 0
    right = n - 1
    
    # Check increasing part 1 to k
    for i in range(1, k + 1):
        while left < n and A[left] < i:
            left += 1
        if left == n:
            return False
        left += 1
    
    # Check decreasing part k-1 to 1
    for i in range(1, k):
        while right >= 0 and A[right] < i:
            right -= 1
        if right < 0:
            return False
        right -= 1
    
    return left <= right + 1

def max_pyramid_size(N, A):
    low, high = 1, (N + 1) // 2
    best_k = 1
    
    while low <= high:
        mid = (low + high) // 2
        if can_form_pyramid(A, mid):
            best_k = mid
            low = mid + 1
        else:
            high = mid - 1
    
    return best_k

if __name__ == "__main__":
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    A = list(map(int, data[1:]))
    
    result = max_pyramid_size(N, A)
    print(result)