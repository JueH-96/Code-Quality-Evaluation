# YOUR CODE HERE
n = int(input())
p = list(map(int, input().split()))
q = int(input())

pos = {}
for i in range(n):
    pos[p[i]] = i + 1

for _ in range(q):
    a, b = map(int, input().split())
    if pos[a] < pos[b]:
        print(a)
    else:
        print(b)