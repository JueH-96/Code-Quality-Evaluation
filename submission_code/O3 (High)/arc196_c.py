import sys

MOD = 998244353

def main() -> None:
    input_data = sys.stdin.read().strip().split()
    if not input_data:
        return
    N = int(input_data[0])
    S = input_data[1].strip()

    # Necessary conditions
    if S[0] != 'B' or S[-1] != 'W':
        print(0)
        return

    blacks = whites = 0
    answer = 1

    for ch in S:
        if ch == 'B':
            blacks += 1
        else:  # ch == 'W'
            diff = blacks - whites         # blacks still unmatched so far
            if diff <= 0:                  # prefix would contain no unmatched black
                print(0)
                return
            answer = (answer * diff) % MOD
            whites += 1

    # after scanning we must have used exactly N blacks and N whites
    # This is guaranteed by the input, but we keep the check for completeness.
    if blacks != N or whites != N:
        print(0)
        return

    print(answer % MOD)

if __name__ == "__main__":
    main()