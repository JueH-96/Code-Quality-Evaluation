import sys
from collections import deque
import sys

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N, M = int(input[ptr]), int(input[ptr+1])
    ptr +=2

    sets = []
    element_to_set_indices = dict()  # Maps each element to list of set indices
    found = False

    for i in range(N):
        A = int(input[ptr])
        ptr +=1
        elements = list(map(int, input[ptr:ptr+A]))
        ptr +=A
        sets.append(elements)
        unique_elements = set(elements)
        if 1 in unique_elements and M in unique_elements:
            found = True
        # Update element_to_set_indices
        for e in unique_elements:
            if e not in element_to_set_indices:
                element_to_set_indices[e] = []
            element_to_set_indices[e].append(i)

    if found:
        print(0)
        return

    # Initialize BFS
    INF = float('inf')
    distance = [INF] * (M + 1)
    distance[1] = 0
    q = deque([1])

    # Processed sets tracking
    processed_sets = [INF] * N

    while q:
        e = q.popleft()
        if e not in element_to_set_indices:
            continue  # No sets containing e, skip
        for set_idx in element_to_set_indices[e]:
            if processed_sets[set_idx] <= distance[e]:
                continue
            processed_sets[set_idx] = distance[e]
            # Iterate all elements in the set
            for f in sets[set_idx]:
                new_dist = distance[e] + 1
                if new_dist < distance[f]:
                    distance[f] = new_dist
                    q.append(f)

    if distance[M] == INF:
        print(-1)
    else:
        print(distance[M] - 1)

if __name__ == '__main__':
    main()