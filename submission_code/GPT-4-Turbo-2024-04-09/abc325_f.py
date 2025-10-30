def solve():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    D = list(map(int, data[1:N+1]))
    L1, C1, K1 = map(int, data[N+1:N+4])
    L2, C2, K2 = map(int, data[N+4:N+7])
    
    # Sort sensors by cost efficiency (cost per meter)
    sensors = sorted([(L1, C1, K1), (L2, C2, K2)], key=lambda x: x[1]/x[0])
    L1, C1, K1 = sensors[0]
    L2, C2, K2 = sensors[1]
    
    total_cost = 0
    
    for length in D:
        # Try to cover each section with minimum cost
        cost = float('inf')
        best_combination = None
        
        # Try all combinations of using up to K1 sensors of type 1 and up to K2 sensors of type 2
        for k1 in range(min(K1, length // L1 + 1) + 1):
            remaining_length = length - k1 * L1
            if remaining_length <= 0:
                # If k1 sensors of type 1 are enough to cover the section
                current_cost = k1 * C1
                if current_cost < cost:
                    cost = current_cost
                    best_combination = (k1, 0)
            else:
                # Calculate how many type 2 sensors are needed
                k2 = (remaining_length + L2 - 1) // L2
                if k2 <= K2:
                    current_cost = k1 * C1 + k2 * C2
                    if current_cost < cost:
                        cost = current_cost
                        best_combination = (k1, k2)
        
        if best_combination is None or cost == float('inf'):
            print(-1)
            return
        
        # Update total cost and reduce the available sensors
        k1, k2 = best_combination
        total_cost += k1 * C1 + k2 * C2
        K1 -= k1
        K2 -= k2
    
    print(total_cost)