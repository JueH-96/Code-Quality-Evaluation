# YOUR CODE HERE
def solve():
    n = int(input())
    a = list(map(int, input().split()))
    
    results = []
    
    for k in range(n):
        # Create a list of (index, size) for active slimes
        slimes = [(i, a[i]) for i in range(n)]
        takahashi_idx = k
        takahashi_size = a[k]
        
        # Keep absorbing until no more possible
        changed = True
        while changed:
            changed = False
            
            # Find Takahashi's current position in the active slimes list
            current_pos = -1
            for i, (idx, size) in enumerate(slimes):
                if idx == takahashi_idx:
                    current_pos = i
                    break
            
            # Check left neighbor
            if current_pos > 0:
                left_idx, left_size = slimes[current_pos - 1]
                if left_size < takahashi_size:
                    takahashi_size += left_size
                    slimes.pop(current_pos - 1)
                    changed = True
                    continue
            
            # Check right neighbor
            if current_pos < len(slimes) - 1:
                right_idx, right_size = slimes[current_pos + 1]
                if right_size < takahashi_size:
                    takahashi_size += right_size
                    slimes.pop(current_pos + 1)
                    changed = True
                    continue
        
        results.append(takahashi_size)
    
    print(' '.join(map(str, results)))

solve()