def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    H = list(map(int, input_data[1:]))

    # p[j] = index of the nearest building to the left of j that is taller than building j
    # If none is taller, p[j] = 0
    p = [0]*(N+1)       # We'll use 1-based indexing for convenience
    stack = []

    # Compute p[] using a monotonic stack (Next Greater Element to the left)
    for j in range(1, N+1):
        # Pop until we find a taller building on the stack or stack is empty
        while stack and H[stack[-1]-1] <= H[j-1]:
            stack.pop()
        p[j] = stack[-1] if stack else 0
        stack.append(j)

    # We'll use a difference array "diff" to mark increments
    # diff[i] will eventually tell us how many times c[i] should be incremented
    diff = [0]*(N+2)  # A bit larger to safely handle boundary

    for j in range(1, N+1):
        # For building j, it contributes +1 to all i in [p[j], j-1]
        # If p[j] = 0, we use 1 as the start because building indices start at 1
        if j == 1:
            # No i < 1
            continue
        L = p[j] if p[j] != 0 else 1
        R = j - 1
        if L <= R:
            diff[L] += 1
            diff[R+1] -= 1

    # Now compute the prefix sums of diff to get the final counts
    c = [0]*(N+1)
    running_sum = 0
    for i in range(1, N+1):
        running_sum += diff[i]
        c[i] = running_sum

    # Print results for i = 1..N
    print(" ".join(map(str, c[1:]))) 

# Do not forget to call main()
if __name__ == "__main__":
    main()