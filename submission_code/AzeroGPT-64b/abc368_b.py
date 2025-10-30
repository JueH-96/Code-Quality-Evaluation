n = int(input())
a = list(map(int, input().split()))
ans = a.count(0)
a = list(map(lambda x:x-1, a))
for i in a:
    if i >= 0:
        ans += 1
print(ans)