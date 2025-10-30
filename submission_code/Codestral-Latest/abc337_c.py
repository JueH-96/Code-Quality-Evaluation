import sys
from collections import deque

input = sys.stdin.read
data = input().split()

N = int(data[0])
A = list(map(int, data[1:]))

# Create a list to store the order of people
order = [-1] * N
# Create a list to store the number of people standing in front of each person
count = [0] * N

# Initialize the queue with people who are at the front of the line
queue = deque()
for i in range(N):
    if A[i] == -1:
        queue.append(i)
        order[i] = i

# Process the queue to determine the order of people
while queue:
    person = queue.popleft()
    for i in range(N):
        if A[i] == person + 1:
            count[i] += 1
            if count[i] == A[i]:
                order[i] = i
                queue.append(i)

# Print the order of people
print(" ".join(map(lambda x: str(x + 1), order)))