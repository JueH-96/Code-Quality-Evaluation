import sys

def main() -> None:
    N = int(sys.stdin.readline().strip())

    # The first 10 palindromes are the one–digit numbers 0 … 9
    if N <= 10:
        print(N - 1)
        return

    # remove the one–digit block
    N -= 10

    length = 2                     # current length we are looking at
    while True:
        half_len = (length + 1) // 2          # length of the determining prefix

        # how many palindromes have this length
        if length == 1:                        # (we never reach this branch –
            cnt = 10                           #  kept just for completeness)
        else:
            cnt = 9 * pow(10, half_len - 1)

        if N > cnt:
            N -= cnt
            length += 1
            continue

        # it is inside this length
        idx = N - 1                            # 0-based index in this block
        prefix_start = pow(10, half_len - 1)   # smallest valid prefix
        prefix = prefix_start + idx            # the prefix we need

        s = str(prefix)
        if length % 2:                         # odd length
            palindrome = s + s[:-1][::-1]
        else:                                  # even length
            palindrome = s + s[::-1]

        print(palindrome)
        return

if __name__ == "__main__":
    main()