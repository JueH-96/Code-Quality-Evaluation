N, M = map(int, input().split())
A = list(map(int, input().split()))
B = list(map(int, input().split()))

# Sort people by their requirements in descending order
B.sort(reverse=True)

# Create a list of (cost, candy_count) for boxes and sort by cost
boxes = sorted([(A[i], A[i]) for i in range(N)])

# For each requirement, find the cheapest available box that satisfies it
available = [True] * N
total_cost = 0

for req in B:
    # Find the cheapest available box that satisfies this requirement
    found = False
    for i in range(N):
        if available[i] and boxes[i][1] >= req:
            total_cost += boxes[i][0]
            available[i] = False
            found = True
            break
    
    if not found:
        total_cost = -1
        break

print(total_cost)