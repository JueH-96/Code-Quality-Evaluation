# YOUR CODE HERE
def max_pyramid_size(N, A):
    left, right = 0, N - 1
    max_size = 0
    current_size = 0
    current_peak = 0

    while left <= right:
        if A[left] < current_size + 1 and A[right] < current_size + 1:
            break
        
        if A[left] >= A[right]:
            if A[left] >= current_size + 1:
                current_size += 1
                current_peak = max(current_peak, current_size)
                left += 1
            else:
                current_size -= 1
        else:
            if A[right] >= current_size + 1:
                current_size += 1
                current_peak = max(current_peak, current_size)
                right -= 1
            else:
                current_size -= 1

    return current_peak

N = int(input())
A = list(map(int, input().split()))

print(max_pyramid_size(N, A))