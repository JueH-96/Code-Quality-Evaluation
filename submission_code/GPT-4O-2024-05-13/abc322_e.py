# YOUR CODE HERE
import itertools

def can_achieve_goal(N, K, P, plans):
    min_cost = float('inf')
    found = False
    
    for r in range(1, N + 1):
        for combination in itertools.combinations(plans, r):
            cost = sum(plan[0] for plan in combination)
            parameters = [0] * K
            for plan in combination:
                for j in range(K):
                    parameters[j] += plan[j + 1]
            if all(param >= P for param in parameters):
                min_cost = min(min_cost, cost)
                found = True
    
    return min_cost if found else -1

# Read input
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
K = int(data[1])
P = int(data[2])

plans = []
index = 3
for i in range(N):
    C_i = int(data[index])
    A_i = list(map(int, data[index + 1:index + 1 + K]))
    plans.append([C_i] + A_i)
    index += 1 + K

# Calculate and print the result
result = can_achieve_goal(N, K, P, plans)
print(result)