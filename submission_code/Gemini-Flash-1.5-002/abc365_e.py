# YOUR CODE HERE
n = int(input())
a = list(map(int, input().split()))

ans = 0
for i in range(n - 1):
    xor_sum = 0
    for j in range(i, n):
        xor_sum ^= a[j]
        ans += xor_sum

print(ans)