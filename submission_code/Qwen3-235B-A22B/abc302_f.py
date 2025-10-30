import sys
from collections import deque

def main():
    input = sys.stdin.read().split()
    ptr = 0

    N = int(input[ptr])
    ptr += 1
    M = int(input[ptr])
    ptr += 1

    found = False
    sets = []
    element_to_sets = [[] for _ in range(M + 1)]
    for i in range(N):
        A_i = int(input[ptr])
        ptr += 1
        elements = list(map(int, input[ptr:ptr + A_i]))
        ptr += A_i

        has_1 = 1 in elements
        has_m = M in elements
        if has_1 and has_m:
            found = True

        sets.append(elements)
        for e in elements:
            element_to_sets[e].append(i)

    if found:
        print(0)
        return

    # BFS setup
    distance = [float('inf')] * (M + 1)
    distance[1] = 0
    visited_sets = [False] * len(sets)
    q = deque([1])

    while q:
        u = q.popleft()
        if u == M:
            break
        for set_idx in element_to_sets[u]:
            if not visited_sets[set_idx]:
                visited_sets[set_idx] = True
                for v in sets[set_idx]:
                    if distance[v] > distance[u] + 1:
                        distance[v] = distance[u] + 1
                        q.append(v)

    if distance[M] == float('inf'):
        print(-1)
    else:
        print(distance[M] - 1)

if __name__ == '__main__':
    main()