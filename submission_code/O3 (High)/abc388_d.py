import sys


def main() -> None:
    input_ = sys.stdin.readline

    N = int(input_())
    A = [0] + list(map(int, input_().split()))          # 1-based indexing

    expire = [0] * (N + 3)                              # expire[t]: donors that stop before step t
    active = 0                                          # current number of adults that still have ≥1 stone
    B = [0] * (N + 1)                                   # answer array (1-based)

    for i in range(1, N + 1):
        # remove donors whose stones became 0 in the previous year
        active -= expire[i]

        gifts = active                                  # stones received from older adults
        stones_now = A[i] + gifts                       # stones after receiving gifts

        future_years = N - i                            # how many more aliens will grow up
        donate_cnt = stones_now if stones_now <= future_years else future_years

        B[i] = stones_now - donate_cnt                  # stones after all N years

        if donate_cnt:                                  # this alien will act as a donor
            active += 1
            end_step = i + donate_cnt + 1               # first year he is no longer a donor
            if end_step <= N:
                expire[end_step] += 1

    sys.stdout.write(' '.join(map(str, B[1:])) + '
')


if __name__ == "__main__":
    main()