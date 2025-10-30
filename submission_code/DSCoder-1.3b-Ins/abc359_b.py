N = int(input())
A = list(map(int, input().split()))

count = 0
for i in range(1, N+1):
    if A.count(i) == 2 and A.count(i-1) == 1 and A.count(i+1) == 1:
        count += 1

print(count)