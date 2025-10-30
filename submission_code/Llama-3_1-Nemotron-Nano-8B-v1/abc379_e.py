n = int(input())
s = input().strip()

sum_10 = [0] * n
sum_10[-1] = 1
for k in range(n-2, -1, -1):
    sum_10[k] = sum_10[k+1] * 10 + 1

total = 0
for i in range(n):
    digit = int(s[i])
    total += digit * (i + 1) * sum_10[i]

print(total)