def check_valid_sequences(seqs, N):
    # Check if each position has all numbers from 1 to N exactly once
    for pos in range(3):
        nums = set()
        for seq in seqs:
            nums.add(seq[pos])
        if len(nums) != N or any(x > N or x < 1 for x in nums):
            return False
    return True

def get_rank(seq, all_seqs):
    # Get lexicographical rank of a sequence among all sequences
    rank = 1
    for other in all_seqs:
        if other < seq:
            rank += 1
    return rank

def solve():
    MOD = 998244353
    N = int(input())
    A = list(map(int, input().split()))
    B = list(map(int, input().split()))
    
    # Generate all possible permutations for each position
    perms = []
    for i in range(N):
        perm = []
        for j in range(3):
            perm.append(0)
        perms.append(perm)
    
    def generate_sequences(pos, used1, used2, used3):
        if pos == N:
            # Check if sequences are valid
            if not check_valid_sequences(perms, N):
                return 0
            
            # Get all sequences and their reverses
            all_seqs = []
            a = []
            b = []
            
            # Add original and reversed sequences
            for i in range(N):
                seq = tuple(perms[i])
                rev = tuple(reversed(perms[i]))
                all_seqs.append(seq)
                all_seqs.append(rev)
                
                # Check for duplicates
                if len(set(all_seqs)) != len(all_seqs):
                    return 0
                
            all_seqs.sort()
            
            # Get ranks for each sequence
            for i in range(N):
                seq = tuple(perms[i])
                rev = tuple(reversed(perms[i]))
                a.append(get_rank(seq, all_seqs))
                b.append(get_rank(rev, all_seqs))
            
            # Check if matches with given A and B
            for i in range(N):
                if a[i] != A[i]:
                    return 0
                if B[i] != -1 and b[i] != B[i]:
                    return 0
            
            return 1
        
        count = 0
        for i in range(1, N+1):
            if i not in used1:
                for j in range(1, N+1):
                    if j not in used2:
                        for k in range(1, N+1):
                            if k not in used3:
                                perms[pos][0] = i
                                perms[pos][1] = j
                                perms[pos][2] = k
                                count = (count + generate_sequences(
                                    pos + 1,
                                    used1 | {i},
                                    used2 | {j},
                                    used3 | {k}
                                )) % MOD
        return count
    
    result = generate_sequences(0, set(), set(), set())
    print(result)

solve()