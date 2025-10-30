# YOUR CODE HERE
n = int(input())
ans = 0
for i in range(2, int(n + 1)):
    count = 0
    temp = n
    while temp > 0:
        count += temp // i
        temp //= i
    ans += count * (i - 1)
print(ans)