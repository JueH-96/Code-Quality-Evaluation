def main():
    import sys
    data = sys.stdin.read().strip().split()
    N, M = map(int, data)

    solutions = []

    def backtrack(seq, idx, start):
        # If we've chosen N elements, store the sequence
        if idx == N:
            solutions.append(seq[:])
            return
        # Otherwise, try all valid next elements
        for x in range(start, M + 1):
            seq.append(x)
            # For the next step, the difference must be at least 10
            # so the next start should be x + 10
            backtrack(seq, idx + 1, x + 10)
            seq.pop()

    # For the first element, it can be from 1 to M
    # Then subsequent elements must follow the difference constraint
    backtrack([], 0, 1)

    # Print results
    print(len(solutions))
    for seq in solutions:
        print(' '.join(map(str, seq)))


# Do not forget to call main, otherwise no points will be awarded
main()