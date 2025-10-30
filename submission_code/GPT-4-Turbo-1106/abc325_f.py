from math import ceil

def solve(N, D, L1, C1, K1, L2, C2, K2):
    # Sort sensors by cost efficiency (cost per meter)
    sensors = sorted([(L1, C1, K1), (L2, C2, K2)], key=lambda x: x[1]/x[0])
    total_cost = 0

    for section_length in D:
        cost = float('inf')
        for i in range(K1 + 1):
            # Check if using i sensors of type 1 is possible
            if i * L1 >= section_length:
                cost = min(cost, i * C1)
                break
            # Calculate remaining length to be covered by sensors of type 2
            remaining_length = section_length - i * L1
            # Calculate number of sensors of type 2 needed
            sensors_needed = ceil(remaining_length / L2)
            # Check if it is possible to use the calculated number of sensors of type 2
            if sensors_needed <= K2:
                cost = min(cost, i * C1 + sensors_needed * C2)
        
        # If it is impossible to cover the section, return -1
        if cost == float('inf'):
            return -1
        total_cost += cost
    
    return total_cost

# Read input
N = int(input().strip())
D = list(map(int, input().strip().split()))
L1, C1, K1 = map(int, input().strip().split())
L2, C2, K2 = map(int, input().strip().split())

# Solve the problem and print the answer
answer = solve(N, D, L1, C1, K1, L2, C2, K2)
print(answer)