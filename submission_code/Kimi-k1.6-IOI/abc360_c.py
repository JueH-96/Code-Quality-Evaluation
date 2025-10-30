n = int(input())
A = list(map(int, input().split()))
W = list(map(int, input().split()))

total = sum(W)
max_weights = [0] * (n + 1)  # Using 1-based indexing for boxes

for box, weight in zip(A, W):
    if weight > max_weights[box]:
        max_weights[box] = weight

sum_max = sum(max_weights[1:n+1])
print(total - sum_max)