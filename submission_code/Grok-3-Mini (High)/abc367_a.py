import sys
data = sys.stdin.read().split()
A = int(data[0])
B = int(data[1])
C = int(data[2])

if C > B:
    asleep = B <= A and A <= C - 1
else:  # C < B
    asleep = (A >= B) or (A <= C - 1)

if not asleep:
    print("Yes")
else:
    print("No")