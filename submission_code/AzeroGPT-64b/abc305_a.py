n = int(input())
ans = (n // 5) * 5
if n % 5 >= 2.5:
  ans += 5

print(ans)