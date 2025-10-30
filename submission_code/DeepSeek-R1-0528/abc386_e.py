import itertools

def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    
    n = int(data[0])
    k = int(data[1])
    A = list(map(int, data[2:2+n]))
    
    if k == n:
        total_xor = 0
        for a in A:
            total_xor ^= a
        print(total_xor)
    else:
        total_xor_val = 0
        for a in A:
            total_xor_val ^= a
            
        m = min(k, n - k)
        use_complement = (k > n - k)
        
        max_val = 0
        for comb in itertools.combinations(A, m):
            xor_set = 0
            for x in comb:
                xor_set ^= x
            if use_complement:
                candidate = total_xor_val ^ xor_set
            else:
                candidate = xor_set
            if candidate > max_val:
                max_val = candidate
        print(max_val)

if __name__ == "__main__":
    main()