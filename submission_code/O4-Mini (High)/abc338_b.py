def main():
    import sys
    S = sys.stdin.readline().strip()
    # count occurrences of each lowercase letter
    counts = [0] * 26
    for ch in S:
        counts[ord(ch) - ord('a')] += 1
    # find the maximum count
    max_count = max(counts)
    # among letters with max_count, pick the earliest alphabetically
    for i in range(26):
        if counts[i] == max_count:
            print(chr(ord('a') + i))
            return

# call main to execute
main()