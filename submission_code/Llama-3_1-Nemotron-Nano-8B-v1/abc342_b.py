n = int(input())
p = list(map(int, input().split()))
position = {num: i+1 for i, num in enumerate(p)}
q = int(input())
for _ in range(q):
    a, b = map(int, input().split())
    if position[a] < position[b]:
        print(a)
    else:
        print(b)