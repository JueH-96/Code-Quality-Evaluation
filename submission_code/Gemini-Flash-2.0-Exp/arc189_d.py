def solve():
    n = int(input())
    a = list(map(int, input().split()))
    
    results = []
    for k in range(n):
        max_size = 0
        
        def calculate_max_size(current_a, current_k):
            nonlocal max_size
            max_size = max(max_size, current_a[current_k])
            
            
            can_absorb = False
            
            # Try absorbing left
            if current_k > 0 and current_a[current_k-1] < current_a[current_k]:
                can_absorb = True
                new_a = current_a[:]
                new_size = new_a[current_k] + new_a[current_k-1]
                del new_a[current_k-1]
                
                calculate_max_size(new_a, max(0, current_k-1))
            
            # Try absorbing right
            if current_k < len(current_a) - 1 and current_a[current_k+1] < current_a[current_k]:
                can_absorb = True
                new_a = current_a[:]
                new_size = new_a[current_k] + new_a[current_k+1]
                del new_a[current_k+1]
                
                calculate_max_size(new_a, current_k)
            
            if not can_absorb:
                max_size = max(max_size, current_a[current_k])
        
        calculate_max_size(a[:], k)
        results.append(max_size)
    
    print(*results)

solve()