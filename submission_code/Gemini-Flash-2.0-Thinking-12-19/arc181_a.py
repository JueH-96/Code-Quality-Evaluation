def solve():
    n = int(input())
    p = list(map(int, input().split()))
    
    if all(p[i] == i + 1 for i in range(n)):
        print(0)
        return
        
    initial_permutation = tuple(p)
    queue = [(initial_permutation, 0)]
    visited_permutations = {initial_permutation}
    
    while queue:
        current_permutation_tuple, operations_count = queue.pop(0)
        current_permutation = list(current_permutation_tuple)
        
        if all(current_permutation[i] == i + 1 for i in range(n)):
            print(operations_count)
            return
            
        for k in range(1, n + 1):
            next_permutation = list(current_permutation)
            if k >= 2:
                segment_to_sort = next_permutation[:k-1]
                segment_to_sort.sort()
                next_permutation[:k-1] = segment_to_sort
            if k <= n - 1:
                segment_to_sort = next_permutation[k:]
                segment_to_sort.sort()
                next_permutation[k:] = segment_to_sort
                
            next_permutation_tuple = tuple(next_permutation)
            if next_permutation_tuple not in visited_permutations:
                visited_permutations.add(next_permutation_tuple)
                queue.append((next_permutation_tuple, operations_count + 1))
                
t = int(input())
for _ in range(t):
    solve()