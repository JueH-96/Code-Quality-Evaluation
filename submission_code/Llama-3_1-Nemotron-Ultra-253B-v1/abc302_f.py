import sys
from collections import deque
from sys import stdin
from collections import defaultdict

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N, M = int(input[ptr]), int(input[ptr+1])
    ptr +=2

    element_to_sets = defaultdict(list)
    sets = []
    contains_1 = set()
    contains_M = set()

    for i in range(N):
        A_i = int(input[ptr])
        ptr +=1
        elements = list(map(int, input[ptr:ptr+A_i]))
        ptr += A_i
        s = set(elements)
        sets.append(s)
        for x in s:
            element_to_sets[x].append(i)
        if 1 in s:
            contains_1.add(i)
        if M in s:
            contains_M.add(i)

    # Check if any set contains both 1 and M
    for i in range(N):
        if 1 in sets[i] and M in sets[i]:
            print(0)
            return

    # Check if sources and targets are non-empty
    if not contains_1 or not contains_M:
        print(-1)
        return

    # BFS
    visited = [False] * N
    queue = deque()
    for s in contains_1:
        visited[s] = True
        queue.append( (s, 0) )

    processed_elements = set()
    found = False

    while queue:
        current_set, dist = queue.popleft()
        if current_set in contains_M:
            print(dist)
            found = True
            break
        # Process all elements in current_set that haven't been processed yet
        for x in sets[current_set]:
            if x not in processed_elements:
                processed_elements.add(x)
                # Iterate through all sets that contain x
                for neighbor in element_to_sets[x]:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        queue.append( (neighbor, dist + 1) )

    if not found:
        print(-1)

if __name__ == "__main__":
    main()