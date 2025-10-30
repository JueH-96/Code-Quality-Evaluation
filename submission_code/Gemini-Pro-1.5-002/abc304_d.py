# YOUR CODE HERE
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

min_strawberries = float('inf')
max_strawberries = float('-inf')

a = [0] + a + [W]
b = [0] + b + [H]

for i in range(A + 1):
    for j in range(B + 1):
        count = 0
        for p, q in strawberries:
            if a[i] < p < a[i+1] and b[j] < q < b[j+1]:
                count += 1
        min_strawberries = min(min_strawberries, count)
        max_strawberries = max(max_strawberries, count)

print(min_strawberries, max_strawberries)