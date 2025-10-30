# YOUR CODE HERE
import sys
from collections import deque

input = sys.stdin.read().split()
N = int(input[0])
A = list(map(int, input[1:N+1]))
Q = int(input[N+1])
queries = input[N+2:]

# Use a dictionary to keep track of the next element for quick access
next_elem = {}
prev_elem = {}

# Initialize the next and previous elements
for i in range(N):
    if i + 1 < N:
        next_elem[A[i]] = A[i + 1]
    if i - 1 >= 0:
        prev_elem[A[i]] = A[i - 1]

# Use a deque to maintain the order of elements
A_deque = deque(A)

for i in range(Q):
    query = queries[i * 2]
    x = int(queries[i * 2 + 1])
    
    if query == '1':
        y = int(queries[i * 2 + 2])
        if x in next_elem:
            next_x = next_elem[x]
            next_elem[x] = y
            prev_elem[y] = x
            if next_x in prev_elem:
                prev_elem[next_x] = y
            next_elem[y] = next_x
        else:
            # x is the last element
            prev_elem[y] = x
            next_elem[x] = y
        A_deque.appendleft(y)  # Add y to the deque, order will be corrected later
    elif query == '2':
        if x in next_elem:
            next_x = next_elem[x]
            if x in prev_elem:
                prev_x = prev_elem[x]
                next_elem[prev_x] = next_x
                prev_elem[next_x] = prev_x
            else:
                # x is the first element
                prev_elem[next_x] = None
        else:
            # x is the last element
            if x in prev_elem:
                prev_x = prev_elem[x]
                next_elem[prev_x] = None
            else:
                # x is the only element, which is not possible as per constraints
                pass
        A_deque.remove(x)

# Reconstruct the list in the correct order
result = []
current = A_deque[0]
while current is not None:
    result.append(current)
    current = next_elem.get(current, None)

print(" ".join(map(str, result)))