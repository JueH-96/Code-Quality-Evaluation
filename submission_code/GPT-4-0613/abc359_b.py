N = int(input())
A = list(map(int, input().split()))
count = 0
for i in range(1, N+1):
    if A.index(i)+2 == A[A.index(i)+1:].index(i)+A.index(i)+1:
        count += 1
print(count)