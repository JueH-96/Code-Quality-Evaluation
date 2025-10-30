import sys
data = sys.stdin.read().strip()
A, B, C = map(int, data.split())

if (A == B + C) or (B == A + C) or (C == A + B) or (A == B and A == C):
    print("Yes")
else:
    print("No")