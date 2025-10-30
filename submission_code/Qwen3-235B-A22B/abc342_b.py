n = int(input())
p_list = list(map(int, input().split()))
pos = {p: i+1 for i, p in enumerate(p_list)}
q = int(input())
for _ in range(q):
    a, b = map(int, input().split())
    if pos[a] < pos[b]:
        print(a)
    else:
        print(b)