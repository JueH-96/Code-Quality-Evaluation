# YOUR CODE HERE
def max_kagamimochi(N, A):
    used = [False] * N  # Keep track of which mochi have been used
    kagami_count = 0
    
    for i in range(N):
        if used[i]:
            continue
        
        # Try to find the smallest mochi that can be the bottom for the current mochi
        for j in range(i + 1, N):
            if not used[j] and A[i] * 2 <= A[j]:
                used[i] = True
                used[j] = True
                kagami_count += 1
                break
    
    return kagami_count

N = int(input())
A = list(map(int, input().split()))

print(max_kagamimochi(N, A))