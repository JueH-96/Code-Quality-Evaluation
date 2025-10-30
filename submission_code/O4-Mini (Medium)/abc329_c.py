import sys

def main():
    input = sys.stdin.readline
    N = int(input())
    S = input().rstrip()
    # max_run[c] will store the maximum run length of character c in S
    max_run = [0] * 26

    prev_char = ''
    current_run = 0

    for ch in S:
        if ch == prev_char:
            current_run += 1
        else:
            # finish the previous run
            if prev_char:
                idx = ord(prev_char) - ord('a')
                if current_run > max_run[idx]:
                    max_run[idx] = current_run
            prev_char = ch
            current_run = 1

    # account for the last run
    if prev_char:
        idx = ord(prev_char) - ord('a')
        if current_run > max_run[idx]:
            max_run[idx] = current_run

    # the number of distinct single-character repetitions is
    # the sum over all characters of their maximum run length
    print(sum(max_run))

if __name__ == "__main__":
    main()