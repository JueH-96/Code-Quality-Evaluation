def main():
    import sys
    sys.setrecursionlimit(10000)
    N_M = sys.stdin.read().strip().split()
    N = int(N_M[0])
    M = int(N_M[1])

    sequences = []

    def generate(seq, position):
        if position == N:
            sequences.append(seq.copy())
            return
        if position == 0:
            min_val = 1
        else:
            min_val = seq[position - 1] + 10
        max_val = M - 10 * (N - position - 1)
        for A_i in range(min_val, max_val + 1):
            seq.append(A_i)
            generate(seq, position + 1)
            seq.pop()

    generate([], 0)

    X = len(sequences)
    print(X)
    for seq in sequences:
        print(' '.join(map(str, seq)))

if __name__ == "__main__":
    main()