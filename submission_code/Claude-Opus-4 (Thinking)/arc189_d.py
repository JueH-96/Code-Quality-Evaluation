# YOUR CODE HERE
N = int(input())
A = list(map(int, input().split()))

B = []

for K in range(N):
    left = K
    right = K
    current_sum = A[K]
    
    changed = True
    while changed:
        changed = False
        
        # Try to expand left
        if left > 0 and A[left-1] < current_sum:
            left -= 1
            current_sum += A[left]
            changed = True
        
        # Try to expand right
        if right < N-1 and A[right+1] < current_sum:
            right += 1
            current_sum += A[right]
            changed = True
    
    B.append(current_sum)

print(' '.join(map(str, B)))