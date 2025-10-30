N = int(input())
A = list(map(int, input().split()))
A.sort()
count = 0
while len(A) > 1:
    A[-1] -= 1
    A[-2] -= 1
    if A[-1] == 0:
        A.pop()
    if A[-2] == 0:
        A.pop()
    A.sort()
    count += 1
print(count)