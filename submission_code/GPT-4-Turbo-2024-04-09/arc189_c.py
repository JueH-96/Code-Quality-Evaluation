import sys
from collections import deque

input = sys.stdin.read
data = input().split()

N = int(data[0])
X = int(data[1]) - 1  # Convert to zero-based index

A = list(map(int, data[2:N+2]))
B = list(map(int, data[N+2:2*N+2]))
P = list(map(lambda x: int(x) - 1, data[2*N+2:3*N+2]))
Q = list(map(lambda x: int(x) - 1, data[3*N+2:4*N+2]))

# To solve the problem, we need to determine if we can isolate all balls to box X
# We need to reverse the process, i.e., see if all balls can be traced back to box X

# We will use a BFS approach to trace back from each box to see if we can reach box X
# We will use two visited arrays to keep track of visited nodes for red and blue balls

visited_red = [False] * N
visited_blue = [False] * N
queue = deque()

# Start BFS from box X since we want to see if all balls can end up here
if A[X] > 0:
    visited_red[X] = True
    queue.append((X, 'red'))
if B[X] > 0:
    visited_blue[X] = True
    queue.append((X, 'blue'))

# BFS to trace back
while queue:
    current, color = queue.popleft()
    
    # Check all boxes to see if they can send balls to the current box
    for i in range(N):
        if color == 'red' and P[i] == current and not visited_red[i]:
            visited_red[i] = True
            queue.append((i, 'red'))
        if color == 'blue' and Q[i] == current and not visited_blue[i]:
            visited_blue[i] = True
            queue.append((i, 'blue'))

# Check if all balls are traceable to box X
all_traceable = True
operations_count = 0

for i in range(N):
    if i != X:
        if A[i] > 0 and not visited_red[i]:
            all_traceable = False
        if B[i] > 0 and not visited_blue[i]:
            all_traceable = False
    if visited_red[i]:
        operations_count += 1
    if visited_blue[i]:
        operations_count += 1

# If not all balls are traceable to box X, it's impossible
if not all_traceable:
    print(-1)
else:
    # The number of operations is the number of times we've marked a box as visited
    # Since each box can be visited twice (once for red and once for blue), we need to adjust
    print(operations_count)