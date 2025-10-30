import sys
input = sys.stdin.read

def solve():
    data = input().split()
    N = int(data[0])
    X = int(data[1])
    
    U = []
    D = []
    for i in range(N):
        U.append(int(data[2 + 2*i]))
        D.append(int(data[3 + 2*i]))
    
    # Calculate the sum of each pair
    H_values = [U[i] + D[i] for i in range(N)]
    
    # We need to find a target H such that all U_i + D_i can be made equal to H
    # and the difference between consecutive U_i is at most X.
    
    # The minimum possible H is max(U_i + D_i) because we can only decrease lengths.
    min_H = max(H_values)
    
    # We need to check if it's possible to make all U_i + D_i = min_H
    # while ensuring |U_i - U_{i+1}| <= X
    
    # To minimize the cost, we should try to reduce the teeth lengths as little as needed.
    # We will try to find the smallest H such that all conditions are satisfied.
    
    # We will use binary search on H to find the minimum cost to make all U_i + D_i = H
    # and |U_i - U_{i+1}| <= X.
    
    # The maximum possible H is sum of max(U_i) + max(D_i)
    max_H = max(U) + max(D)
    
    def can_fit(H):
        # We need to check if we can make all U_i + D_i = H
        # and adjust U_i such that |U_i - U_{i+1}| <= X
        
        # We will use a greedy approach to adjust U_i within the bounds
        # that are allowed by the constraints of U_i + D_i = H and |U_i - U_{i+1}| <= X
        
        # Start with the first tooth
        current_U = U[0]
        for i in range(1, N):
            # Calculate the next U_i based on the current U_i and the constraint |U_i - U_{i+1}| <= X
            next_U_min = max(0, current_U - X)
            next_U_max = current_U + X
            
            # Also, U_{i+1} needs to satisfy U_{i+1} + D_{i+1} = H
            # Thus, U_{i+1} = H - D_{i+1}
            target_U = H - D[i]
            
            # The next U_i should be within [next_U_min, next_U_max] and also needs to be target_U
            if target_U < next_U_min or target_U > next_U_max:
                return False
            
            # Update current_U to the target_U for the next iteration
            current_U = target_U
        
        return True
    
    # Binary search for the minimum H that works
    low, high = min_H, max_H
    while low < high:
        mid = (low + high) // 2
        if can_fit(mid):
            high = mid
        else:
            low = mid + 1
    
    # The minimum H that works is now in `low`
    # Calculate the cost to make all U_i + D_i = low
    min_cost = 0
    for i in range(N):
        current_sum = U[i] + D[i]
        if current_sum > low:
            min_cost += current_sum - low
    
    print(min_cost)