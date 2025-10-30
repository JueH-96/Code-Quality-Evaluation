def main():
    import sys
    N = int(sys.stdin.readline())
    A = list(map(int, sys.stdin.readline().split()))
    M = int(sys.stdin.readline())
    B = list(map(int, sys.stdin.readline().split()))
    L = int(sys.stdin.readline())
    C = list(map(int, sys.stdin.readline().split()))
    Q = int(sys.stdin.readline())
    X = list(map(int, sys.stdin.readline().split()))
    
    # Precompute all a + b sums
    ab_sums = []
    for a in A:
        for b in B:
            ab_sums.append(a + b)
    
    # Compute all possible a + b + c sums and store in a set
    total_sums = set()
    for s in ab_sums:
        for c in C:
            total_sums.add(s + c)
    
    # Process each query
    result = []
    for x in X:
        if x in total_sums:
            result.append("Yes")
        else:
            result.append("No")
    
    print('
'.join(result))

if __name__ == "__main__":
    main()