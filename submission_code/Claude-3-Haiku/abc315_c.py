from collections import defaultdict

N = int(input())
flavors = defaultdict(list)
for i in range(N):
    F, S = map(int, input().split())
    flavors[F].append(S)

max_satisfaction = 0
for flavor, deliciousness in flavors.items():
    deliciousness.sort(reverse=True)
    for i in range(0, len(deliciousness), 2):
        s, t = deliciousness[i], deliciousness[i+1] if i+1 < len(deliciousness) else 0
        if s != t:
            max_satisfaction = max(max_satisfaction, s + t)
        else:
            max_satisfaction = max(max_satisfaction, s + t//2)

print(max_satisfaction)