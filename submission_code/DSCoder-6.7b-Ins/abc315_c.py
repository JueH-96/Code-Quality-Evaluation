import sys

N = int(sys.stdin.readline())

flavors = []
for _ in range(N):
    F, S = map(int, sys.stdin.readline().split())
    flavors.append((F, S))

flavors.sort(key=lambda x: (x[1], -x[0]))

max_satisfaction = 0
for i in range(N-1):
    F1, S1 = flavors[i]
    for j in range(i+1, N):
        F2, S2 = flavors[j]
        if F1 != F2:
            satisfaction = S1 + S2
        else:
            satisfaction = S1 + S2/2
        max_satisfaction = max(max_satisfaction, satisfaction)

print(int(max_satisfaction))