import sys
from collections import deque

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N, M = int(input[ptr]), int(input[ptr+1])
    ptr += 2
    
    sets = []
    has_1 = [False] * N
    has_M = [False] * N
    element_to_sets = [[] for _ in range(M + 1)]  # elements are 1-based
    
    for i in range(N):
        A_i = int(input[ptr])
        ptr += 1
        elements = list(map(int, input[ptr:ptr+A_i]))
        ptr += A_i
        
        contains_1 = 1 in elements
        contains_M = M in elements
        has_1[i] = contains_1
        has_M[i] = contains_M
        
        sets.append(elements)
        
        for x in elements:
            element_to_sets[x].append(i)
    
    # Check if any set already contains both 1 and M
    for i in range(N):
        if has_1[i] and has_M[i]:
            print(0)
            return
    
    # BFS setup
    q = deque()
    dist_set = [-1] * N
    dist_element = [-1] * (M + 1)
    
    # Initialize with sets containing 1
    for i in range(N):
        if has_1[i]:
            dist_set[i] = 0
            q.append(i)
    
    # Process the queue
    while q:
        val = q.popleft()
        if val < N:
            # It's a set, process its elements
            for x in sets[val]:
                if dist_element[x] == -1:
                    dist_element[x] = dist_set[val] + 1
                    q.append(x)
        else:
            # It's an element, process its sets
            x = val
            for set_i in element_to_sets[x]:
                if dist_set[set_i] == -1:
                    dist_set[set_i] = dist_element[x] + 1
                    if has_M[set_i]:
                        print(dist_set[set_i] // 2)
                        return
                    q.append(set_i)
    
    # If no path found
    print(-1)

if __name__ == "__main__":
    main()