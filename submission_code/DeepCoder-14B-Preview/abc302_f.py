from collections import deque, defaultdict
import sys

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    M = int(input[ptr])
    ptr += 1

    sets_elements = []
    element_to_sets = defaultdict(list)

    for _ in range(N):
        A_i = int(input[ptr])
        ptr += 1
        elements = set(map(int, input[ptr:ptr+A_i]))
        ptr += A_i
        sets_elements.append(elements)
        for e in elements:
            element_to_sets[e].append(len(sets_elements) - 1)  # 0-based index

    # Precompute which sets contain 1 or M
    has_one = [False] * N
    has_m = [False] * N
    for i in range(N):
        if 1 in sets_elements[i]:
            has_one[i] = True
        if M in sets_elements[i]:
            has_m[i] = True

    # Check if any set contains both 1 and M
    for i in range(N):
        if has_one[i] and has_m[i]:
            print(0)
            return

    # Build the adjacency list
    adj = [[] for _ in range(N)]
    for i in range(N):
        seen = set()
        elements_i = sets_elements[i]
        for e in elements_i:
            for neighbor in element_to_sets[e]:
                if neighbor == i:
                    continue
                if neighbor not in seen:
                    adj[i].append(neighbor)
                    seen.add(neighbor)

    # BFS setup
    distance = [-1] * N
    queue = deque()

    for i in range(N):
        if has_one[i]:
            distance[i] = 0
            queue.append(i)

    # Perform BFS
    while queue:
        u = queue.popleft()
        for v in adj[u]:
            if distance[v] == -1:
                distance[v] = distance[u] + 1
                if has_m[v]:
                    print(distance[v])
                    return
                queue.append(v)

    # If no path found
    print(-1)

if __name__ == '__main__':
    main()