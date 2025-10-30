import math
N = int(input().strip())
s = set()
for a in range(2, math.ceil(math.pow(N, 0.5)) + 1):
    b = 2
    while True:
        x = pow(a, b)
        if x > N:
            break
        s.add(x)
        b += 1
print(len(s))