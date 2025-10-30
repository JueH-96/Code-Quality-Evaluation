def solve():
    n = int(input())
    a = list(map(int, input().split()))
    b = list(map(int, input().split()))
    
    mod = 998244353
    
    def count_valid_sequences(perms):
        s = []
        t = []
        for p in perms:
            s.append(tuple(p))
            t.append(tuple(p[::-1]))
        
        all_seq = sorted(s + t)
        
        if len(all_seq) != len(set(all_seq)):
            return 0
        
        seq_a = []
        seq_b = []
        
        for i in range(n):
            seq_a.append(all_seq.index(s[i]) + 1)
            seq_b.append(all_seq.index(t[i]) + 1)
            
        if seq_a == a:
            
            valid = True
            for i in range(n):
                if b[i] != -1 and b[i] != seq_b[i]:
                    valid = False
                    break
            if valid:
                return 1
        return 0

    import itertools
    
    count = 0
    
    
    all_perms = list(itertools.permutations(range(1, n + 1)))
    
    
    
    
    for p1 in all_perms:
        for p2 in all_perms:
            for p3 in all_perms:
                perms = []
                for i in range(n):
                    perms.append([p1[i], p2[i], p3[i]])
                count = (count + count_valid_sequences(perms)) % mod
    
    print(count)

solve()