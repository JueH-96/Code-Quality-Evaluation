# YOUR CODE HERE
import sys
from collections import defaultdict

input = sys.stdin.read().split()
Q = int(input[0])
index = 1
bag = defaultdict(int)
unique_count = 0
results = []

while index < len(input):
    query_type = input[index]
    if query_type == '1':
        x = int(input[index + 1])
        if bag[x] == 0:
            unique_count += 1
        bag[x] += 1
        index += 2
    elif query_type == '2':
        x = int(input[index + 1])
        bag[x] -= 1
        if bag[x] == 0:
            unique_count -= 1
        index += 2
    elif query_type == '3':
        results.append(unique_count)
        index += 1

for result in results:
    print(result)