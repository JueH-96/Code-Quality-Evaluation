N = int(input())
W = []
X = []
for _ in range(N):
    w, x = map(int, input().split())
    W.append(w)
    X.append(x)

max_sum = 0
for T in range(24):
    current_sum = 0
    for i in range(N):
        local_start = (T + X[i]) % 24
        if 9 <= local_start < 18:
            current_sum += W[i]
    if current_sum > max_sum:
        max_sum = current_sum

print(max_sum)