N = int(input())
D = list(map(int, input().split()))
count = 0
for i in range(1, N+1):
    if i < 10:
        count += 1
        if D[i-1] >= 11:
            count += 1
print(count)