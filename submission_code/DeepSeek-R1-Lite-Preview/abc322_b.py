N, M = map(int, input().split())
S = input().strip()
T = input().strip()

prefix = (T[:N] == S)
suffix = (T[-N:] == S)

if prefix and suffix:
    print(0)
elif prefix:
    print(1)
elif suffix:
    print(2)
else:
    print(3)