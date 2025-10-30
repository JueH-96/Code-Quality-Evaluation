# YOUR CODE HERE
n, t = map(int, input().split())
s = input()
x = list(map(int, input().split()))

ans = 0
for i in range(n):
    for j in range(i + 1, n):
        if s[i] == '0' and s[j] == '1':
            if x[i] > x[j]:
                if x[i] - x[j] <= 2 * t + 0.2:
                    ans += 1
        elif s[i] == '1' and s[j] == '0':
            if x[j] > x[i]:
                if x[j] - x[i] <= 2 * t + 0.2:
                    ans += 1

print(ans)