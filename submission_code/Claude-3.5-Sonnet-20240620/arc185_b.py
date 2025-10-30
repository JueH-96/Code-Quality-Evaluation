# YOUR CODE HERE
def can_make_non_decreasing(N, A):
    total_sum = sum(A)
    expected_sum = sum(range(A[0], A[0] + N))
    
    if total_sum < expected_sum:
        return "No"
    
    current_min = A[0]
    current_sum = 0
    for i in range(N):
        current_sum += A[i]
        expected_sum = (i + 1) * current_min + sum(range(i + 1))
        if current_sum < expected_sum:
            return "No"
        current_min = max(current_min, A[i])
    
    return "Yes"

T = int(input())
for _ in range(T):
    N = int(input())
    A = list(map(int, input().split()))
    print(can_make_non_decreasing(N, A))