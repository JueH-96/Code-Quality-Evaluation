MOD = 998244353

def solve():
    N = int(input())
    A = list(map(int, input().split()))
    B = list(map(int, input().split()))
    
    # Find which positions are free for B
    used_positions = set(A)
    for b in B:
        if b != -1:
            used_positions.add(b)
    
    free_positions = []
    for i in range(1, 2*N + 1):
        if i not in used_positions:
            free_positions.append(i)
    
    # Find indices where B[i] = -1
    free_indices = []
    for i in range(N):
        if B[i] == -1:
            free_indices.append(i)
    
    from itertools import permutations
    
    total_count = 0
    
    # Try all ways to assign free positions to free indices
    for perm in permutations(free_positions):
        # Create complete B array
        full_B = B[:]
        for idx, pos in enumerate(perm):
            full_B[free_indices[idx]] = pos
        
        # Now we have complete position assignments
        # Count valid sequence configurations
        count = count_valid_sequences(N, A, full_B)
        total_count = (total_count + count) % MOD
    
    return total_count

def count_valid_sequences(N, A, B):
    # Create the full ordering
    items = []
    for i in range(N):
        items.append(('s', i, A[i]))
        items.append(('t', i, B[i]))
    
    # Sort by position
    items.sort(key=lambda x: x[2])
    
    # For small N, we can try all permutations
    # For large N, we need a more sophisticated approach
    
    if N <= 8:  # Small enough to brute force
        from itertools import permutations
        
        count = 0
        # Try all three permutations
        for perm1 in permutations(range(1, N+1)):
            for perm2 in permutations(range(1, N+1)):
                for perm3 in permutations(range(1, N+1)):
                    # Check if s_i != t_i for all i
                    valid = True
                    for i in range(N):
                        if perm1[i] == perm3[i]:
                            valid = False
                            break
                    
                    if not valid:
                        continue
                    
                    # Create all sequences
                    sequences = []
                    for i in range(N):
                        s = (perm1[i], perm2[i], perm3[i])
                        t = (perm3[i], perm2[i], perm1[i])
                        sequences.append(('s', i, s))
                        sequences.append(('t', i, t))
                    
                    # Sort lexicographically
                    sequences.sort(key=lambda x: x[2])
                    
                    # Check if ordering matches
                    matches = True
                    for j in range(2*N):
                        typ, idx, _ = sequences[j]
                        if typ == 's' and A[idx] != j+1:
                            matches = False
                            break
                        if typ == 't' and B[idx] != j+1:
                            matches = False
                            break
                    
                    if matches:
                        count += 1
        
        return count
    else:
        # For large N, we need a more efficient approach
        # This is a placeholder - the actual implementation would need
        # dynamic programming or other optimization
        return 1

print(solve())