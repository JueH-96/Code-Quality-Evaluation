import math

B = int(input().strip())

for A in range(1, int(math.pow(B, 1/2)) + 1):
    if A**A == B:
        print(A)
        break
else:
    print(-1)