n = int(input())
s = list(map(int, input().strip()))

sum0 = [0] * (n + 2)
sum1 = [0] * (n + 2)

sum0[n] = 0
sum1[n] = 1

for i in range(n-1, 0, -1):
    a_next = s[i]
    sum0[i] = sum1[i+1]
    if a_next == 1:
        sum1[i] = 1 + sum0[i+1]
    else:
        sum1[i] = 1 + sum1[i+1]

total = 0
for i in range(1, n+1):
    if s[i-1] == 0:
        total += sum0[i]
    else:
        total += sum1[i]

print(total)