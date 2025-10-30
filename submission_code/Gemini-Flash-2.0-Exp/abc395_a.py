N = int(input())
A = list(map(int, input().split()))

increasing = True
for i in range(N - 1):
    if A[i] >= A[i+1]:
        increasing = False
        break

if increasing:
    print("Yes")
else:
    print("No")