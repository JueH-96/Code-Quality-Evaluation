import sys

# Read input
W, H = map(int, input().split())
N = int(input())
strawberries = []
for _ in range(N):
    p, q = map(int, input().split())
    strawberries.append((p, q))
A = int(input())
a = list(map(int, input().split()))
B = int(input())
b = list(map(int, input().split()))

# Sort the cut lines
a.sort()
b.sort()

# Count the number of strawberries in each piece
min_strawberries = float('inf')
max_strawberries = 0
for i in range(len(a) + 1):
    for j in range(len(b) + 1):
        x1 = 0 if i == 0 else a[i-1]
        x2 = W if i == len(a) else a[i]
        y1 = 0 if j == 0 else b[j-1]
        y2 = H if j == len(b) else b[j]
        count = 0
        for p, q in strawberries:
            if x1 < p < x2 and y1 < q < y2:
                count += 1
        min_strawberries = min(min_strawberries, count)
        max_strawberries = max(max_strawberries, count)

# Print the answer
print(min_strawberries, max_strawberries)