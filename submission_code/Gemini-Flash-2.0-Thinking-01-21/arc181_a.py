def solve():
    n = int(input())
    p = list(map(int, input().split()))
    initial_permutation = tuple(p)
    target_permutation = tuple(range(1, n + 1))
    if initial_permutation == target_permutation:
        print(0)
        return
    
    queue = [(initial_permutation, 0)]
    visited_permutations = {initial_permutation}
    
    while queue:
        current_permutation, operations_count = queue.pop(0)
        if current_permutation == target_permutation:
            print(operations_count)
            return
            
        for k in range(1, n + 1):
            next_permutation_list = list(current_permutation)
            if k >= 2:
                prefix = next_permutation_list[0:k-1]
                prefix.sort()
                next_permutation_list[0:k-1] = prefix
            if k <= n - 1:
                suffix = next_permutation_list[k:n]
                suffix.sort()
                next_permutation_list[k:n] = suffix
            next_permutation = tuple(next_permutation_list)
            
            if next_permutation not in visited_permutations:
                visited_permutations.add(next_permutation)
                queue.append((next_permutation, operations_count + 1))
                
    return -1 # Should not reach here based on problem statement

t = int(input())
for _ in range(t):
    solve()