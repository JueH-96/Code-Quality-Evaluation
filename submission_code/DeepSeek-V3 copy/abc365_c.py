def find_max_x(N, M, A):
    A_sorted = sorted(A)
    total = 0
    for i in range(N):
        if total + A_sorted[i] <= M:
            total += A_sorted[i]
        else:
            break
    else:
        return "infinite"
    
    low = 0
    high = A_sorted[-1]
    result = 0
    
    while low <= high:
        mid = (low + high) // 2
        current_total = 0
        for a in A_sorted:
            if a <= mid:
                current_total += a
            else:
                current_total += mid
        if current_total <= M:
            result = mid
            low = mid + 1
        else:
            high = mid - 1
    return result

# Read input
N, M = map(int, input().split())
A = list(map(int, input().split()))

# Compute and print the result
result = find_max_x(N, M, A)
print(result)