n = int(input())
A = list(map(int, input().split()))
W = list(map(int, input().split()))

max_w = [0] * (n + 1)  # 1-based indexing for boxes

for i in range(n):
    box = A[i]
    weight = W[i]
    if weight > max_w[box]:
        max_w[box] = weight

sum_max = sum(max_w[1:n+1])
total = sum(W) - sum_max

print(total)