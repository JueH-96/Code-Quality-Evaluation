def solve():
    N = int(input())
    A = list(map(int, input().split()))
    B = list(map(int, input().split()))
    
    MOD = 998244353
    
    # This is a complex constraint satisfaction problem
    # We need to use more sophisticated techniques
    
    # For now, let's implement a direct but optimized approach
    from itertools import permutations
    
    def check_valid_arrangement(perm1, perm2, perm3):
        # Create sequences
        seqs = []
        for i in range(N):
            seqs.append((perm1[i], perm2[i], perm3[i]))
        
        # Create all sequences with their reverses
        all_seqs = []
        for i, seq in enumerate(seqs):
            all_seqs.append((seq, i, False))
            all_seqs.append((seq[::-1], i, True))
        
        # Sort lexicographically
        all_seqs.sort()
        
        # Check for duplicates
        for i in range(len(all_seqs) - 1):
            if all_seqs[i][0] == all_seqs[i+1][0]:
                return False, None, None
        
        # Get ranks
        a = [0] * N
        b = [0] * N
        
        for rank, (seq, idx, is_reverse) in enumerate(all_seqs, 1):
            if is_reverse:
                b[idx] = rank
            else:
                a[idx] = rank
        
        return True, a, b
    
    count = 0
    
    # Generate all permutations for the three positions
    for perm1 in permutations(range(1, N+1)):
        for perm2 in permutations(range(1, N+1)):
            for perm3 in permutations(range(1, N+1)):
                valid, a, b = check_valid_arrangement(perm1, perm2, perm3)
                
                if not valid:
                    continue
                
                # Check constraints
                if a != A:
                    continue
                
                valid_b = True
                for i in range(N):
                    if B[i] != -1 and b[i] != B[i]:
                        valid_b = False
                        break
                
                if valid_b:
                    count += 1
    
    return count % MOD

print(solve())