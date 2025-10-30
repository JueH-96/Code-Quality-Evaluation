def solve():
    n, k = map(int, input().split())
    
    import itertools
    
    def is_good(seq, n, k):
        if len(seq) != n * k:
            return False
        counts = {}
        for x in seq:
            if x < 1 or x > n:
                return False
            counts[x] = counts.get(x, 0) + 1
        
        if len(counts) != n:
            return False
        
        for i in range(1, n + 1):
            if counts.get(i, 0) != k:
                return False
        return True

    
    all_seqs = []
    
    possible_values = list(range(1, n + 1))
    
    
    
    all_permutations = itertools.permutations(list(range(1, n + 1)) * k)
    
    good_seqs = []
    for perm in all_permutations:
        if is_good(perm, n, k):
            good_seqs.append(list(perm))
            
    
    good_seqs.sort()
    
    
    index = (len(good_seqs) + 1) // 2 - 1
    
    print(*good_seqs[index])

solve()