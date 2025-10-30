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
    right = 10**18  # A large enough upper bound
    
    result = 0
    
    while left <= right:
        mid = (left + right) // 2
        total_cost = 0
        possible = True
        
        for i in range(N):
            min_cost = float('inf')
            # Try all possible combinations of S_i and T_i to achieve at least mid
            # Since the number of machines can be large, we need a smarter way
            # We can find the minimal cost to achieve at least mid for each process
            # For S_i: number of S_i machines needed is ceil(mid / A_i)
            # For T_i: number of T_i machines needed is ceil(mid / B_i)
            # Also, we can mix S_i and T_i
            # To find the minimal cost, we can iterate over the number of S_i machines and compute the required T_i machines
            # Since the number of S_i machines can be up to mid // A_i + 1, we can limit the loop
            max_S = (mid // A[i]) + 2
            min_cost_for_process = float('inf')
            for s in range(max_S):
                required_T = max(0, (mid - s * A[i] + B[i] - 1) // B[i])
                cost = s * P[i] + required_T * Q[i]
                if cost < min_cost_for_process:
                    min_cost_for_process = cost
            if min_cost_for_process == float('inf'):
                possible = False
                break
            total_cost += min_cost_for_process
            if total_cost > X:
                possible = False
                break
        
        if possible:
            result = mid
            left = mid + 1
        else:
            right = mid - 1
    
    print(result)

if __name__ == "__main__":
    main()