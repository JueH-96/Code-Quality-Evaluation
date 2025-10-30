def main():
    import sys
    input_data = sys.stdin.read().split()
    idx = 0

    n = int(input_data[idx])
    idx += 1
    A = list(map(int, input_data[idx:idx+n]))
    idx += n

    m = int(input_data[idx])
    idx += 1
    B = list(map(int, input_data[idx:idx+m]))
    idx += m

    l = int(input_data[idx])
    idx += 1
    C = list(map(int, input_data[idx:idx+l]))
    idx += l

    q = int(input_data[idx])
    idx += 1
    X = list(map(int, input_data[idx:idx+q]))

    # Compute the possible sums from A, B, and C by first calculating sums of A and B.
    AB_sums = set()
    for a in A:
        for b in B:
            AB_sums.add(a + b)
    
    possibleSums = set()
    for ab in AB_sums:
        for c in C:
            possibleSums.add(ab + c)

    # Answer each query.
    results = []
    for x in X:
        results.append("Yes" if x in possibleSums else "No")
    
    sys.stdout.write("
".join(results))

if __name__ == '__main__':
    main()