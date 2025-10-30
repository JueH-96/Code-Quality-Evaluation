def main():
    import sys
    data = sys.stdin.read().strip().split()
    N, K = map(int, data[:2])
    R = list(map(int, data[2:]))

    sequence = []

    def backtrack(pos, mod_sum):
        if pos == N:
            # If we've placed N elements, check if sum mod K == 0
            if mod_sum == 0:
                print(" ".join(map(str, sequence)))
            return

        # Generate values from 1 to R[pos]
        for val in range(1, R[pos] + 1):
            sequence.append(val)
            backtrack(pos + 1, (mod_sum + val) % K)
            sequence.pop()

    # Start backtracking from position 0 with sum mod K = 0
    backtrack(0, 0)

# Do NOT forget to call main
if __name__ == "__main__":
    main()