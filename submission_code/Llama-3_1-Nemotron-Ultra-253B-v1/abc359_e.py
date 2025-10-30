n = int(input())
h = list(map(int, input().split()))
ans = []
current = 0
delay = 0
for i in range(n):
    current += h[i] + 1
    current += delay
    ans.append(current)
    delay += 1
print(' '.join(map(str, ans)))