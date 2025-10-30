# YOUR CODE HERE
n = int(input())
a = list(map(int, input().split()))

total = 0
for i in range(n-1):
    xor_val = 0
    for j in range(i, n):
        xor_val ^= a[j]
        if j > i:
            total += xor_val

print(total)