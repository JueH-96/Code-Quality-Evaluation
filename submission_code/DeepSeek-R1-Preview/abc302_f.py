import sys
from collections import deque
from collections import defaultdict

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    M = int(input[ptr])
    ptr += 1

    sets = []
    element_to_sets = defaultdict(list)
    has_both = False

    for i in range(N):
        A_i = int(input[ptr])
        ptr += 1
        elements = list(map(int, input[ptr:ptr+A_i]))
        ptr += A_i
        sets.append(elements)
        s_set = set(elements)
        if 1 in s_set and M in s_set:
            has_both = True
        for x in elements:
            element_to_sets[x].append(i)

    if has_both:
        print(0)
        return

    INF = float('inf')
    hyper_count = [INF] * (M + 1)
    processed = [False] * N
    queue = deque()

    # Initialize elements in sets containing 1
    for s_idx in element_to_sets.get(1, []):
        s = sets[s_idx]
        for x in s:
            if hyper_count[x] == INF:
                hyper_count[x] = 1
                queue.append(x)

    # BFS
    while queue:
        x = queue.popleft()
        for s_idx in element_to_sets[x]:
            if processed[s_idx]:
                continue
            processed[s_idx] = True
            s = sets[s_idx]
            for y in s:
                if hyper_count[y] > hyper_count[x] + 1:
                    hyper_count[y] = hyper_count[x] + 1
                    queue.append(y)

    if hyper_count[M] == INF:
        print(-1)
    else:
        print(hyper_count[M] - 1)

if __name__ == '__main__':
    main()