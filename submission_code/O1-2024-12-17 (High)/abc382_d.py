def main():
    import sys
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    M = int(data[1])

    solutions = []
    seq = [0]*N

    def backtrack(index, start_val):
        if index == N:
            solutions.append(seq[:])
            return
        # Prune if it's impossible to complete the sequence
        # i.e., we need at least 10*(N - 1 - index) space after the current pick
        for val in range(start_val, M + 1):
            if val > M - 10*(N - 1 - index):
                break
            seq[index] = val
            backtrack(index + 1, val + 10)

    backtrack(0, 1)

    print(len(solutions))
    for s in solutions:
        print(" ".join(map(str, s)))

# Do not forget to call main()
main()