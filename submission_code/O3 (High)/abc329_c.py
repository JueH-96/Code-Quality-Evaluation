import sys

def main() -> None:
    input_data = sys.stdin.read().strip().splitlines()
    if not input_data:
        return
    # N is not actually needed except for potential validation
    # N = int(input_data[0])
    S = input_data[1].rstrip()

    max_run = [0] * 26           # maximum consecutive length for each letter

    current_char = S[0]
    current_len = 1

    for ch in S[1:]:
        if ch == current_char:
            current_len += 1
        else:
            idx = ord(current_char) - 97
            if current_len > max_run[idx]:
                max_run[idx] = current_len
            current_char = ch
            current_len = 1

    # handle the last run
    idx = ord(current_char) - 97
    if current_len > max_run[idx]:
        max_run[idx] = current_len

    # answer is the sum of all maximum run lengths
    print(sum(max_run))

if __name__ == "__main__":
    main()