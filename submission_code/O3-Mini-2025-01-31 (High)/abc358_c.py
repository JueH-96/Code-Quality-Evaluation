def main():
    import sys
    data = sys.stdin.read().split()
    N = int(data[0])
    M = int(data[1])

    # Convert each stand's string to a bitmask representation.
    # Each bit corresponds to one flavor (bit j is 1 if flavor j is sold).
    stands = []
    for i in range(N):
        s = data[i + 2]
        bitmask = 0
        for j in range(M):
            if s[j] == 'o':
                bitmask |= (1 << j)
        stands.append(bitmask)

    # target bitmask where all M flavors are obtained
    target = (1 << M) - 1
    ans = float('inf')

    # There are at most 2^N possible subsets (N <= 10) so it's efficient to iterate.
    # We use bitmasking to represent a set of stands.
    for mask in range(1, 1 << N):
        union = 0
        count = 0
        for i in range(N):
            if mask & (1 << i):
                union |= stands[i]
                count += 1
        if union == target:
            ans = min(ans, count)

    sys.stdout.write(str(ans))

if __name__ == '__main__':
    main()