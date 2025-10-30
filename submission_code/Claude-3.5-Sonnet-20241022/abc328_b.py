def is_repdigit(n):
    if n == 0:
        return True
    n = str(n)
    return len(set(n)) == 1

N = int(input())
days = list(map(int, input().split()))

count = 0
for month in range(1, N+1):
    if not is_repdigit(month):
        continue
        
    for day in range(1, days[month-1]+1):
        if is_repdigit(day):
            count += 1

print(count)