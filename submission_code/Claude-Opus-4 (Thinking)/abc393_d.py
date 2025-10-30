# YOUR CODE HERE
N = int(input())
S = input()

# Find positions of 1s
positions = []
for i in range(N):
    if S[i] == '1':
        positions.append(i)

k = len(positions)

if k <= 1:
    print(0)
else:
    # Compute q[i] = positions[i] - i
    q = []
    for i in range(k):
        q.append(positions[i] - i)
    
    # Find median
    if k % 2 == 1:
        median = q[k // 2]
    else:
        median = q[k // 2 - 1]
    
    # Compute cost
    cost = 0
    for val in q:
        cost += abs(val - median)
    
    print(cost)