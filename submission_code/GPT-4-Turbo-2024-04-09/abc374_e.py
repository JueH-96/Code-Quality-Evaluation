def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    X = int(data[1])
    
    processes = []
    index = 2
    for _ in range(N):
        A = int(data[index])
        P = int(data[index+1])
        B = int(data[index+2])
        Q = int(data[index+3])
        processes.append((A, P, B, Q))
        index += 4
    
    # Binary search for the maximum production capacity
    low, high = 0, 10**10  # We set high to a very large number initially
    
    def can_produce_at_least(capacity):
        # Check if we can achieve at least `capacity` production with budget X
        total_cost = 0
        for A, P, B, Q in processes:
            # Minimum units needed for Si and Ti to achieve at least `capacity`
            units_S = (capacity + A - 1) // A  # Ceiling of capacity / A
            units_T = (capacity + B - 1) // B  # Ceiling of capacity / B
            
            # Cost to achieve at least `capacity` with Si and Ti
            cost_S = units_S * P
            cost_T = units_T * Q
            
            # We take the minimum cost of using either Si or Ti
            min_cost = min(cost_S, cost_T)
            
            total_cost += min_cost
            if total_cost > X:
                return False
        
        return total_cost <= X
    
    # Binary search to find the maximum capacity we can achieve within budget
    while low < high:
        mid = (low + high + 1) // 2
        if can_produce_at_least(mid):
            low = mid
        else:
            high = mid - 1
    
    print(low)

if __name__ == "__main__":
    main()