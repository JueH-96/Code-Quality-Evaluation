# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    X = int(data[1])
    
    A = []
    P = []
    B = []
    Q = []
    
    index = 2
    for _ in range(N):
        A.append(int(data[index]))
        P.append(int(data[index+1]))
        B.append(int(data[index+2]))
        Q.append(int(data[index+3]))
        index += 4
    
    # Binary search for the maximum possible production capacity
    left = 0
    right = 10**18  # A very large number to cover the maximum possible capacity
    
    best = 0
    
    while left <= right:
        mid = (left + right) // 2
        total_cost = 0
        possible = True
        
        for i in range(N):
            # Calculate the minimum cost to achieve at least mid products per day for process i
            min_cost = float('inf')
            
            # Try all possible combinations of S_i and T_i
            # Since the number of machines can be large, we need a smarter way
            # We can iterate over the number of S_i machines and compute the required T_i machines
            # Since the number of S_i machines can be up to mid // A_i + 1, but that's too large
            # So, we need a better approach
            
            # Let's find the minimal cost to achieve at least mid products per day
            # W_i = s * A_i + t * B_i >= mid
            # Cost = s * P_i + t * Q_i
            # We need to minimize s * P_i + t * Q_i under the constraint s * A_i + t * B_i >= mid
            
            # We can iterate s from 0 to (mid // A_i) + 1
            # For each s, compute the minimal t such that t >= (mid - s * A_i) / B_i
            # Then, compute the cost and keep the minimum
            
            min_s = 0
            max_s = (mid // A[i]) + 1 if A[i] != 0 else 0
            min_cost_for_i = float('inf')
            
            for s in range(max_s + 1):
                remaining = mid - s * A[i]
                if remaining <= 0:
                    t = 0
                else:
                    t = (remaining + B[i] - 1) // B[i]
                cost = s * P[i] + t * Q[i]
                if cost < min_cost_for_i:
                    min_cost_for_i = cost
            
            total_cost += min_cost_for_i
            if total_cost > X:
                possible = False
                break
        
        if possible:
            best = mid
            left = mid + 1
        else:
            right = mid - 1
    
    print(best)

if __name__ == "__main__":
    main()