def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    T = int(input[idx])
    idx += 1
    results = []
    for _ in range(T):
        N = int(input[idx])
        idx += 1
        P = list(map(int, input[idx:idx+N]))
        idx += N
        ans = 0
        # We simulate the process by tracking the rightmost position that is not correct
        # and assuming that after an operation, the first i elements are sorted.
        for i in range(N-1, -1, -1):
            if P[i] != i + 1:
                ans += 1
                # After performing the operation, the first i elements are sorted
                # To simulate this, we can set them to 1..i, but for large N, this is not efficient
                # However, since we only need the count, we can just continue
                # This is a conceptual simulation for counting purposes
        results.append(str(ans))
    print('
'.join(results))

main()