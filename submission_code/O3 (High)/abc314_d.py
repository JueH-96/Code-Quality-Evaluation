import sys

def main() -> None:
    input = sys.stdin.readline

    N = int(input())
    S = list(input().strip())
    Q = int(input())

    # time[i] : the step index at which S[i] was last updated by type-1 operation
    time = [-1] * N

    last_global_time = -1      # step index of the latest type-2/3 operation
    case_mode = None           # 'lower', 'upper', or None

    for step in range(Q):
        t, x, c = input().split()
        t = int(t)
        x = int(x)

        if t == 1:
            # single-character update (1-based index)
            S[x - 1] = c
            time[x - 1] = step
        else:
            # whole-string case conversion
            last_global_time = step
            case_mode = 'lower' if t == 2 else 'upper'

    # If no global conversion ever happened, just output current string
    if last_global_time == -1:
        print(''.join(S))
        return

    to_lower = (case_mode == 'lower')
    result = []

    for ch, ts in zip(S, time):
        if ts < last_global_time:          # not updated after the last global op
            ch = ch.lower() if to_lower else ch.upper()
        result.append(ch)

    print(''.join(result))


if __name__ == "__main__":
    main()