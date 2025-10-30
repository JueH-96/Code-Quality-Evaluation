n = int(input())
a = list(map(int, input().split()))

total = 0
for i in range(n-1):
    xor = a[i]
    for j in range(i+1, n):
        xor ^= a[j]
        total += xor

print(total)