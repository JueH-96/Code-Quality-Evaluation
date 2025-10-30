def main():
    import sys
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    S = data[1]

    # Array to store the maximum consecutive-run length we've encountered
    # for each character ('a' -> 0, 'b' -> 1, ..., 'z' -> 25).
    max_run = [0] * 26

    result = 0
    i = 0
    while i < N:
        c = S[i]
        c_idx = ord(c) - ord('a')

        # Count how many identical characters in a row from position i.
        run_len = 1
        j = i + 1
        while j < N and S[j] == c:
            run_len += 1
            j += 1

        # The new repeated-substrings (of character c) contributed by
        # this run are those lengths from (max_run[c] + 1) up to run_len,
        # if run_len is greater than max_run[c].
        new_substrings = run_len - max_run[c_idx]
        if new_substrings > 0:
            result += new_substrings
            max_run[c_idx] = run_len

        i = j

    print(result)

# Don't forget to call main()
main()