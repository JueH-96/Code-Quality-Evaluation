import sys
from collections import deque
from collections import defaultdict

def main():
    input = sys.stdin.read
    data = input().split()
    ptr = 0
    N = int(data[ptr])
    ptr += 1
    M = int(data[ptr])
    ptr += 1

    sets = []
    for _ in range(N):
        A_i = int(data[ptr])
        ptr += 1
        elements = list(map(int, data[ptr:ptr + A_i]))
        ptr += A_i
        sets.append(elements)

    adj = defaultdict(list)
    for idx in range(N):
        for e in sets[idx]:
            adj[e].append(idx)

    distance = [float('inf')] * (M + 1)
    distance[1] = 0
    used = [False] * N
    queue = deque()
    queue.append(1)

    while queue:
        e = queue.popleft()
        for s in adj[e]:
            if not used[s]:
                used[s] = True
                for f in sets[s]:
                    if distance[f] > distance[e] + 1:
                        distance[f] = distance[e] + 1
                        queue.append(f)

    if distance[M] == float('inf'):
        print(-1)
    else:
        print(distance[M] - 1)

if __name__ == '__main__':
    main()