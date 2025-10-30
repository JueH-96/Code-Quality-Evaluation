n = int(input())
a = list(map(int, input().split()))
b = [0] * n
for i, x in enumerate(a):
    b[x - 1] = i if b[x - 1] == 0 else b[x - 1] if a.index(x, b[x - 1] + 1, i) > i else i
print(*sorted(range(n), key=lambda x: b[x]))