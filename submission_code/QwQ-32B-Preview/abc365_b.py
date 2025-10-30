N = int(input())
A = list(map(int, input().split()))

if N < 2:
    print("Error: N should be at least 2")
else:
    max1 = A[0]
    max2 = None
    for i in range(1, N):
        if A[i] > max1:
            max2 = max1
            max1 = A[i]
        elif max2 is None or A[i] > max2:
            max2 = A[i]
    position = A.index(max2) + 1
    print(position)