def main():
    import sys
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    H = list(map(int, data[1:]))

    # We will use the "next greater element" idea:
    # next_greater[i] will store the index j (j > i) of the next building
    # to the right that is taller than building i. If none exists, we set it to N.

    next_greater = [N] * N  # default "no greater element" to point beyond last index
    stack = []

    # Compute next_greater array in O(N) using a decreasing stack (standard approach)
    for i in range(N):
        while stack and H[stack[-1]] < H[i]:
            idx = stack.pop()
            next_greater[idx] = i
        stack.append(i)

    # leaders_count[i] = number of "record-breakers" starting from index i to the end,
    # scanning left to right. Equivalently:
    # leaders_count[i] = 1 + leaders_count[next_greater[i]] if next_greater[i] != N
    #                  = 1 if next_greater[i] = N
    leaders_count = [0] * N
    for i in range(N - 1, -1, -1):
        if next_greater[i] == N:
            leaders_count[i] = 1
        else:
            leaders_count[i] = 1 + leaders_count[next_greater[i]]

    # We want c[i] for i in [0..N-1], which corresponds to "c_{i+1}" in 1-based indexing.
    # c[i] = leaders_count[i+1] if i+1 < N else 0
    c = [0] * N
    for i in range(N - 1):
        c[i] = leaders_count[i + 1]
    c[N - 1] = 0  # the last building has no buildings to the right

    # Print result
    print(" ".join(map(str, c)))

# Do not forget to call main!
if __name__ == "__main__":
    main()