import math
b = int(input())
max_a = int(math.sqrt(b)) + 1
for a in range(1, max_a):
    if a ** a == b:
        print(a)
        break
else:
    print(-1)