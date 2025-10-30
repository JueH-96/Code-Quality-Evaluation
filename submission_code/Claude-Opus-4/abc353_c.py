# YOUR CODE HERE
n = int(input())
a = list(map(int, input().split()))

MOD = 10**8
total = 0

for i in range(n):
    for j in range(i + 1, n):
        total += (a[i] + a[j]) % MOD

print(total)