def main():
    import sys
    input = sys.stdin.read().split()
    ptr = 0

    n = int(input[ptr])
    ptr += 1
    A = list(map(int, input[ptr:ptr + n]))
    ptr += n

    m = int(input[ptr])
    ptr += 1
    B = list(map(int, input[ptr:ptr + m]))
    ptr += m

    l = int(input[ptr])
    ptr += 1
    C = list(map(int, input[ptr:ptr + l]))
    ptr += l

    q = int(input[ptr])
    ptr += 1
    X = list(map(int, input[ptr:ptr + q]))
    ptr += q

    # Precompute all possible sums of A and B
    ab_sums = set()
    for a in A:
        for b in B:
            ab_sums.add(a + b)
    
    # Precompute all possible sums of ab_sums and C
    abc_sums = set()
    for s in ab_sums:
        for c in C:
            abc_sums.add(s + c)
    
    # Process each query
    for x in X:
        if x in abc_sums:
            print("Yes")
        else:
            print("No")

if __name__ == "__main__":
    main()