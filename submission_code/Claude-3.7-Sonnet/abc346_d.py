def min_cost_good_string(S, C):
    N = len(S)
    S = list(S)
    
    # Check if S is already a good string
    count_same = 0
    for i in range(N-1):
        if S[i] == S[i+1]:
            count_same += 1
    
    if count_same == 1:
        return 0
    
    min_cost = float('inf')
    
    for i in range(N-1):
        for same_value in ['0', '1']:
            # Construct a good string where positions i and i+1 have the same value
            T = ['0'] * N
            T[i] = same_value
            T[i+1] = same_value
            
            # Fill the rest of T to ensure all other adjacent pairs are different
            for j in range(i+2, N):
                T[j] = '1' if T[j-1] == '0' else '0'
            
            for j in range(i-1, -1, -1):
                T[j] = '1' if T[j+1] == '0' else '0'
            
            # Calculate cost to transform S to T
            cost = 0
            for j in range(N):
                if S[j] != T[j]:
                    cost += C[j]
            
            min_cost = min(min_cost, cost)
    
    return min_cost

# Read inputs
N = int(input().strip())
S = input().strip()
C = list(map(int, input().strip().split()))

# Compute and print the answer
print(min_cost_good_string(S, C))