import sys
from collections import defaultdict

input = sys.stdin.read
data = input().split()

t = int(data[0])
index = 1
results = []

for _ in range(t):
    S = data[index]
    X = data[index + 1]
    Y = data[index + 2]
    index += 3

    # Count the number of '0's and '1's in X and Y
    count_X = defaultdict(int)
    count_Y = defaultdict(int)

    for char in X:
        count_X[char] += 1

    for char in Y:
        count_Y[char] += 1

    # Check if the counts of '0's and '1's match
    if count_X['0'] == count_Y['0'] and count_X['1'] == count_Y['1']:
        results.append("Yes")
    else:
        results.append("No")

# Print the results
for result in results:
    print(result)