import math

B = int(input())

for A in range(1, int(math.log(B, 2)) + 1):
    if A ** A == B:
        print(A)
        break
else:
    print(-1)