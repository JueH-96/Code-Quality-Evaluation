def main():
    import sys

    data = sys.stdin.read().strip().split()
    N = int(data[0])
    A = list(map(int, data[1:]))

    # pos[x] will hold the current index of the element (x+1) in A.
    pos = [0]*N
    for i in range(N):
        pos[A[i] - 1] = i

    swaps = []

    # For each position i, if it's not correct, swap it with the position
    # of the correct element i+1.
    for i in range(N):
        correct_val = i + 1
        if A[i] != correct_val:
            # p is the position of the element (i+1)
            p = pos[i]

            # Swap in A
            x = A[i]
            y = A[p]
            A[i], A[p] = A[p], A[i]

            # Update pos array
            pos[y - 1] = i
            pos[x - 1] = p

            # Record swap (in 1-based indexing, ensure i < j)
            swaps.append((min(i, p) + 1, max(i, p) + 1))

    # Output result
    print(len(swaps))
    for s in swaps:
        print(s[0], s[1])

# Do not forget to call main!
if __name__ == "__main__":
    main()