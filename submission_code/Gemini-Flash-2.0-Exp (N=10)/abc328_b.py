def is_repdigit(n):
  s = str(n)
  if not s:
    return False
  return len(set(s)) == 1

def solve():
  n = int(input())
  days_in_month = list(map(int, input().split()))
  
  count = 0
  for month in range(1, n + 1):
    for day in range(1, days_in_month[month - 1] + 1):
      if is_repdigit(month) and is_repdigit(day) and str(month)[0] == str(day)[0]:
        count += 1
  print(count)

solve()