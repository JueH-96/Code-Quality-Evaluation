N = int(input())
S = input().strip()

total = 0
for k in range(N):
    digit = int(S[k])
    power = pow(10, N - k)  # 10^(N-k)
    contribution = digit * (k + 1) * (power - 1) // 9
    total += contribution

print(total)