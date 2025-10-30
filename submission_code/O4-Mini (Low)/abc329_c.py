def main():
    import sys
    input = sys.stdin.readline
    
    N = int(input().strip())
    S = input().strip()
    
    # We'll record for each character the maximum length of a contiguous run found.
    max_run = [0] * 26  # one slot per lowercase letter
    
    if N > 0:
        cur_char = S[0]
        cur_len = 1
        # iterate from second character
        for i in range(1, N):
            if S[i] == cur_char:
                cur_len += 1
            else:
                idx = ord(cur_char) - ord('a')
                if cur_len > max_run[idx]:
                    max_run[idx] = cur_len
                cur_char = S[i]
                cur_len = 1
        # finalize the last run
        idx = ord(cur_char) - ord('a')
        if cur_len > max_run[idx]:
            max_run[idx] = cur_len
    
    # The number of distinct substrings that are repeats of one character
    # for each character c is just max_run[c], since we get "c", "cc", ..., "c"*max_run[c].
    answer = sum(max_run)
    print(answer)

if __name__ == "__main__":
    main()