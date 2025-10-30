n = int(input())
d = list(map(int, input().split()))
count = 0
for month in range(1, n+1):
    for day in range(1, d[month-1]+1):
        s = str(month) + str(day)
        if len(set(s)) == 1:
            count +=1
print(count)