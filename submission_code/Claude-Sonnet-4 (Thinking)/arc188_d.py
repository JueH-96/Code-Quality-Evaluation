from itertools import permutations

def solve():
    N = int(input())
    A = list(map(int, input().split()))
    B = list(map(int, input().split()))
    
    MOD = 998244353
    count = 0
    
    # Check for impossible constraints early
    used_positions = set()
    for a in A:
        if a in used_positions:
            return 0
        used_positions.add(a)
    
    for b in B:
        if b != -1:
            if b in used_positions:
                return 0
            used_positions.add(b)
    
    if len(used_positions) > 2*N:
        return 0
    
    # Try all possible column permutations
    for perm1 in permutations(range(1, N+1)):
        for perm2 in permutations(range(1, N+1)):
            for perm3 in permutations(range(1, N+1)):
                # Quick check: sequences can't equal their reverses
                skip = False
                for i in range(N):
                    if perm1[i] == perm3[i]:  # sequence equals its reverse
                        skip = True
                        break
                if skip:
                    continue
                
                # Construct sequences and reverses
                sequences = []
                reverses = []
                for i in range(N):
                    seq = (perm1[i], perm2[i], perm3[i])
                    rev = (perm3[i], perm2[i], perm1[i])
                    sequences.append(seq)
                    reverses.append(rev)
                
                # Check all sequences are distinct
                all_seqs = sequences + reverses
                if len(set(all_seqs)) != 2*N:
                    continue
                
                # Sort and check position constraints
                sorted_seqs = sorted(all_seqs)
                valid = True
                for i in range(N):
                    seq_pos = sorted_seqs.index(sequences[i]) + 1
                    rev_pos = sorted_seqs.index(reverses[i]) + 1
                    
                    if seq_pos != A[i]:
                        valid = False
                        break
                    
                    if B[i] != -1 and rev_pos != B[i]:
                        valid = False
                        break
                
                if valid:
                    count += 1
    
    return count % MOD

print(solve())