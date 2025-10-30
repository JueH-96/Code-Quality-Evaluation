def min_cost_to_fit_teeth(N, X, teeth):
    U = [teeth[i][0] for i in range(N)]
    D = [teeth[i][1] for i in range(N)]
    
    # Calculate the total cost to make the teeth fit well
    total_cost = float('inf')
    
    # We will check for each possible H value
    # H can be in the range [min(U[i] + D[i]) - 1, max(U[i] + D[i])]
    min_H = min(U[i] + D[i] for i in range(N))
    max_H = max(U[i] + D[i] for i in range(N))
    
    for H in range(min_H, max_H + 1):
        current_cost = 0
        
        # Calculate the cost to adjust U and D to fit H
        for i in range(N):
            if U[i] + D[i] < H:
                # Need to increase the sum
                current_cost += H - (U[i] + D[i])
            elif U[i] + D[i] > H:
                # Need to decrease the sum
                current_cost += (U[i] + D[i]) - H
        
        # Check the condition for U_i and U_{i+1}
        valid = True
        for i in range(N - 1):
            if abs(U[i] - U[i + 1]) > X:
                valid = False
                break
        
        if valid:
            total_cost = min(total_cost, current_cost)
    
    return total_cost

import sys
input = sys.stdin.read

def main():
    data = input().strip().splitlines()
    N, X = map(int, data[0].split())
    teeth = [tuple(map(int, line.split())) for line in data[1:N + 1]]
    
    result = min_cost_to_fit_teeth(N, X, teeth)
    print(result)

if __name__ == "__main__":
    main()