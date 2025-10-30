# YOUR CODE HERE
from collections import deque

K = int(input())

queue = deque(range(1, 10))
result = []

while queue:
    n = queue.popleft()
    result.append(n)
    if len(result) >= K:
        print(result[K - 1])
        break
    last_digit = n % 10
    for next_digit in range(last_digit):
        new_n = n * 10 + next_digit
        queue.append(new_n)