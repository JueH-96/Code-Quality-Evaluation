import sys
from collections import deque

def main():
    n, m = map(int, sys.stdin.readline().split())
    sets = []
    sets_with_1 = []
    sets_with_M = []
    element_to_sets = dict()

    for i in range(n):
        a = int(sys.stdin.readline())
        s = set(map(int, sys.stdin.readline().split()))
        sets.append(s)
        contains_1 = 1 in s
        contains_M = m in s
        if contains_1 and contains_M:
            print(0)
            return
        if contains_1:
            sets_with_1.append(i)
        if contains_M:
            sets_with_M.append(i)
        for elem in s:
            if elem not in element_to_sets:
                element_to_sets[elem] = []
            element_to_sets[elem].append(i)
    
    # Check if any set contains both 1 and M after reading all (in case they were added later)
    # Wait, no. Because during reading, we checked each set as it was read. So this is redundant.

    # BFS initialization
    distance = [-1] * n
    q = deque()
    min_distance = float('inf')
    
    for idx in sets_with_1:
        distance[idx] = 1
        q.append(idx)
    
    while q:
        current_set_idx = q.popleft()
        current_dist = distance[current_set_idx]
        current_set = sets[current_set_idx]
        for elem in current_set:
            if elem not in element_to_sets:
                continue
            for neighbor_set_idx in element_to_sets[elem]:
                if distance[neighbor_set_idx] == -1:
                    distance[neighbor_set_idx] = current_dist + 1
                    q.append(neighbor_set_idx)
                    # Check if this neighbor set contains M
                    if neighbor_set_idx in sets_with_M:
                        if distance[neighbor_set_idx] < min_distance:
                            min_distance = distance[neighbor_set_idx]
    
    if min_distance != float('inf'):
        print(min_distance - 1)
    else:
        print(-1)

if __name__ == "__main__":
    main()