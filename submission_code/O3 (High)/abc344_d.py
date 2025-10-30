import sys

INF = 10 ** 9  # a value larger than any possible cost


def read_bag(ai_expected: int, initial_tokens: list, inp) -> list:
    """
    Helper that, given that `ai_expected` strings are expected for the current bag
    and `initial_tokens` already contains some (possibly all) of them, keeps
    reading from `inp` until the whole list is obtained.
    """
    strings = initial_tokens
    while len(strings) < ai_expected:
        strings += inp.readline().strip().split()
    return strings


def main() -> None:
    inp = sys.stdin

    T = inp.readline().strip()
    M = len(T)                          # length of the target
    while True:                         # skip possible blank lines
        n_line = inp.readline()
        if n_line.strip():
            break
    N = int(n_line)

    bags = []
    for _ in range(N):
        # read first non-empty line for this bag
        while True:
            parts = inp.readline().strip().split()
            if parts:
                break
        Ai = int(parts[0])              # number of strings in current bag
        strings_in_line = parts[1:]     # strings that were on the same line
        bag_strings = read_bag(Ai, strings_in_line, inp)
        bags.append(bag_strings)

    # DP: dp[pos] = minimal yen to build prefix T[:pos] after processing current bags
    dp = [INF] * (M + 1)
    dp[0] = 0                           # empty string costs 0

    for bag in bags:
        new_dp = dp[:]                  # choosing "do nothing" keeps cost unchanged
        for pos in range(M + 1):
            if dp[pos] == INF:
                continue
            for s in bag:
                l = len(s)
                if pos + l <= M and T[pos:pos + l] == s:
                    new_cost = dp[pos] + 1
                    if new_cost < new_dp[pos + l]:
                        new_dp[pos + l] = new_cost
        dp = new_dp

    answer = dp[M]
    print(-1 if answer == INF else answer)


if __name__ == "__main__":
    main()