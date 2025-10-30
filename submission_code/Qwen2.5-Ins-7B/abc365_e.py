# YOUR CODE HERE
n = int(input())
a = list(map(int, input().split()))
xor_sum = 0
for i in range(n):
    xor_sum ^= a[i]
ans = 0
for i in range(n):
    ans += a[i] * (xor_sum ^ a[i]) * (i + 1) - xor_sum * (xor_sum ^ a[i]) * (n - i)
print(ans // 2)