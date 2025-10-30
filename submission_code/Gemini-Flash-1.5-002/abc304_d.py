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

cuts_x = [0] + a + [W]
cuts_y = [0] + b + [H]

min_strawberries = float('inf')
max_strawberries = float('-inf')

for i in range(len(cuts_x) - 1):
    for j in range(len(cuts_y) - 1):
        count = 0
        for p, q in strawberries:
            if cuts_x[i] < p < cuts_x[i+1] and cuts_y[j] < q < cuts_y[j+1]:
                count += 1
        min_strawberries = min(min_strawberries, count)
        max_strawberries = max(max_strawberries, count)

print(min_strawberries, max_strawberries)