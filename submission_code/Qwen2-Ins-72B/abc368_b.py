n = int(input())
a = sorted(list(map(int, input().split())), reverse=True)
ans = 0
while len(a) >= 2 and a[1] >= 1:
    a[0] -= 1
    a[1] -= 1
    a = sorted(a, reverse=True)
    ans += 1
print(ans)