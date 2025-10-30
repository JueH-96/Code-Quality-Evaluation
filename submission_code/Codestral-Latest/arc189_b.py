import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
X = list(map(int, data[1:]))

# Calculate the minimum possible sum of the coordinates
min_sum = sum(X[:2]) + sum(X[-2:]) + sum(X[2:-2])

print(min_sum)