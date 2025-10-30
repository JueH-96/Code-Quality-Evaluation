def main() -> None:
    import sys

    # Read N and M
    N_M = sys.stdin.readline().strip().split()
    while len(N_M) < 2:      # in case of trailing spaces/newlines
        N_M += sys.stdin.readline().strip().split()
    N, M = map(int, N_M)

    adjacent_pairs = set()  # store unordered pairs that were ever adjacent

    for _ in range(M):
        line = []
        # Each photo line may be split across several reads if long.
        while len(line) < N:
            line += sys.stdin.readline().strip().split()
        photo = list(map(int, line[:N]))

        for j in range(N - 1):
            a, b = photo[j], photo[j + 1]
            if a > b:
                a, b = b, a        # store as (smaller, larger)
            adjacent_pairs.add((a, b))

    total_pairs = N * (N - 1) // 2
    answer = total_pairs - len(adjacent_pairs)
    print(answer)


if __name__ == "__main__":
    main()