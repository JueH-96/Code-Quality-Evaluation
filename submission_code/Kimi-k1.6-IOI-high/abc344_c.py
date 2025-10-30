def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    ptr = 0
    N = int(data[ptr])
    ptr += 1
    A = list(map(int, data[ptr:ptr+N]))
    ptr += N
    
    M = int(data[ptr])
    ptr += 1
    B = list(map(int, data[ptr:ptr+M]))
    ptr += M
    
    L = int(data[ptr])
    ptr += 1
    C = list(map(int, data[ptr:ptr+L]))
    ptr += L
    
    Q = int(data[ptr])
    ptr += 1
    X = list(map(int, data[ptr:ptr+Q]))
    ptr += Q
    
    # Precompute all possible sums
    sums = set()
    for a in A:
        for b in B:
            for c in C:
                sums.add(a + b + c)
    
    # Check each query
    results = ["Yes" if x in sums else "No" for x in X]
    print('
'.join(results))

if __name__ == "__main__":
    main()