from collections import defaultdict

def solve():
    N, M = map(int, input().split())
    sets = []
    for _ in range(N):
        A = int(input())
        S = list(map(int, input().split()))
        sets.append(set(S))
    
    # Create a dictionary to store the sets that contain each element
    element_sets = defaultdict(list)
    for i, s in enumerate(sets):
        for x in s:
            element_sets[x].append(i)
    
    # Check if 1 and M are in the same set
    if 1 in element_sets and M in element_sets and any(i in element_sets[1] for i in element_sets[M]):
        return 0
    
    # Perform the operations
    operations = 0
    while True:
        # Find two sets with a common element
        found = False
        for x in range(1, M+1):
            if x in element_sets and len(element_sets[x]) > 1:
                i, j = element_sets[x][:2]
                found = True
                break
        if not found:
            return -1
        
        # Merge the two sets
        new_set = sets[i] | sets[j]
        for y in new_set:
            element_sets[y].append(N)
        sets[N] = new_set
        element_sets[x].remove(i)
        element_sets[x].remove(j)
        element_sets[x].append(N)
        N += 1
        operations += 1
        
        # Check if 1 and M are in the same set
        if 1 in element_sets and M in element_sets and any(i in element_sets[1] for i in element_sets[M]):
            return operations