a = input().split()
ans = 0
for i in range(64):
  ans += int(a[i]) * (2 ** (63 - i))
print(ans)