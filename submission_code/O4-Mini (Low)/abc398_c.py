def main():
    import sys
    input = sys.stdin.readline

    N = int(input().strip())
    A = list(map(int, input().split()))

    # Count frequency of each integer
    freq = {}
    for x in A:
        freq[x] = freq.get(x, 0) + 1

    # Find the unique element with the maximum value
    best_val = -1
    best_idx = -1
    for idx, x in enumerate(A, start=1):
        if freq[x] == 1 and x > best_val:
            best_val = x
            best_idx = idx

    # Output the result
    if best_idx == -1:
        print(-1)
    else:
        print(best_idx)

if __name__ == "__main__":
    main()