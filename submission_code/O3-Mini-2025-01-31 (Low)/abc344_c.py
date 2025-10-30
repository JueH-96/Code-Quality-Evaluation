def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    it = iter(input_data)
    
    N = int(next(it))
    A = [int(next(it)) for _ in range(N)]
    M = int(next(it))
    B = [int(next(it)) for _ in range(M)]
    L = int(next(it))
    C = [int(next(it)) for _ in range(L)]
    Q = int(next(it))
    X = [int(next(it)) for _ in range(Q)]
    
    ab_sums = []
    for a in A:
        for b in B:
            ab_sums.append(a+b)
    
    possible_sums = set()
    for s in ab_sums:
        for c in C:
            possible_sums.add(s+c)
    
    out_lines = []
    for x in X:
        if x in possible_sums:
            out_lines.append("Yes")
        else:
            out_lines.append("No")
    
    sys.stdout.write("
".join(out_lines))
    
if __name__ == '__main__':
    main()