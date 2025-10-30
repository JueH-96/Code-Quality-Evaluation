def can_form_pyramid(A, k):
    # Pyramid sequence for size k: 1,2,...,k-1,k,k-1,...,2,1
    pyramid = []
    for i in range(1, k+1):
        pyramid.append(i)
    for i in range(k-1, 0, -1):
        pyramid.append(i)
    
    n = len(A)
    m = len(pyramid)  # 2k-1
    
    # Try all possible starting positions
    for start in range(n - m + 1):
        valid = True
        for i in range(m):
            if A[start + i] < pyramid[i]:
                valid = False
                break
        if valid:
            return True
    
    return False

# Read input
n = int(input())
A = list(map(int, input().split()))

# Binary search for maximum k
left, right = 1, (n + 1) // 2
answer = 1

while left <= right:
    mid = (left + right) // 2
    if can_form_pyramid(A, mid):
        answer = mid
        left = mid + 1
    else:
        right = mid - 1

print(answer)