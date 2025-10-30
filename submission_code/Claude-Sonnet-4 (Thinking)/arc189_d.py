n = int(input())
A = list(map(int, input().split()))

results = []

for k in range(n):
    alive = [True] * n
    pos = k
    size = A[k]
    
    while True:
        absorbed = False
        
        # Find left neighbor
        left_pos = pos - 1
        while left_pos >= 0 and not alive[left_pos]:
            left_pos -= 1
        
        # Find right neighbor
        right_pos = pos + 1
        while right_pos < n and not alive[right_pos]:
            right_pos += 1
        
        # Try to absorb left neighbor
        if left_pos >= 0 and A[left_pos] < size:
            size += A[left_pos]
            alive[left_pos] = False
            absorbed = True
        # Try to absorb right neighbor
        elif right_pos < n and A[right_pos] < size:
            size += A[right_pos]
            alive[right_pos] = False
            absorbed = True
        
        if not absorbed:
            break
    
    results.append(size)

print(' '.join(map(str, results)))