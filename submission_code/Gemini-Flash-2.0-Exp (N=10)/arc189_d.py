def solve():
    n = int(input())
    a = list(map(int, input().split()))
    
    results = []
    for k in range(n):
        max_size = 0
        
        def calculate_max_size(current_a, current_k):
            nonlocal max_size
            max_size = max(max_size, current_a[current_k])
            
            
            left_neighbor = current_k - 1
            right_neighbor = current_k + 1
            
            
            absorbed = False
            
            if left_neighbor >= 0 and current_a[left_neighbor] < current_a[current_k]:
                
                new_a = current_a[:left_neighbor] + [current_a[current_k] + current_a[left_neighbor]] + current_a[current_k+1:]
                calculate_max_size(new_a, left_neighbor)
                absorbed = True
            
            if right_neighbor < len(current_a) and current_a[right_neighbor] < current_a[current_k]:
                new_a = current_a[:current_k] + [current_a[current_k] + current_a[right_neighbor]] + current_a[right_neighbor+1:]
                calculate_max_size(new_a, current_k)
                absorbed = True
            
            if not absorbed:
                return
            
        calculate_max_size(a.copy(), k)
        results.append(max_size)
    
    print(*results)

solve()