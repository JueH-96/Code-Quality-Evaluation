A, B = map(int, input().split())

s = set()

s.add(A - (B - A))
s.add(B - (A - B))
s.add((A + B) / 2)

print(len(s))