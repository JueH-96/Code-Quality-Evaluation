# YOUR CODE HERE
n = int(input())
ans = 0
while n > 1:
    if n % 2 == 0:
        ans += n
        n //= 2
    else:
        ans += n
        n = (n + 1) // 2
print(ans)