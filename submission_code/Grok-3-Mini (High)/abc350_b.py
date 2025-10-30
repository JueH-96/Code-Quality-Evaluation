import sys
data = sys.stdin.read().split()
index = 0
N = int(data[index])
index += 1
Q = int(data[index])
index += 1
s = set()
for _ in range(Q):
    T = int(data[index])
    index += 1
    if T in s:
        s.remove(T)
    else:
        s.add(T)
final_teeth = N - len(s)
print(final_teeth)