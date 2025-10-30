def max_production_capacity(N, X, machines):
    def can_achieve_capacity(target_capacity):
        total_cost = 0
        for A_i, P_i, B_i, Q_i in machines:
            # Calculate the minimum cost to achieve at least target_capacity
            cost_s = float('inf')
            cost_t = float('inf')
            
            if target_capacity <= A_i:
                cost_s = P_i  # We need at least one S_i machine
            else:
                num_s = (target_capacity + A_i - 1) // A_i  # Ceiling division
                cost_s = num_s * P_i
            
            if target_capacity <= B_i:
                cost_t = Q_i  # We need at least one T_i machine
            else:
                num_t = (target_capacity + B_i - 1) // B_i  # Ceiling division
                cost_t = num_t * Q_i
            
            total_cost += min(cost_s, cost_t)
            if total_cost > X:  # Early exit if we exceed budget
                return False
        
        return total_cost <= X

    left, right = 0, 10**9  # Set a reasonable upper bound for capacity
    best_capacity = 0
    
    while left <= right:
        mid = (left + right) // 2
        if can_achieve_capacity(mid):
            best_capacity = mid
            left = mid + 1
        else:
            right = mid - 1
            
    return best_capacity

import sys
input = sys.stdin.read

def main():
    data = input().strip().splitlines()
    first_line = data[0].split()
    N = int(first_line[0])
    X = int(first_line[1])
    
    machines = []
    for i in range(1, N + 1):
        A, P, B, Q = map(int, data[i].split())
        machines.append((A, P, B, Q))
    
    result = max_production_capacity(N, X, machines)
    print(result)

if __name__ == "__main__":
    main()