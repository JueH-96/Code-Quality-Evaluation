def main():
    import sys

    S = sys.stdin.readline().strip()

    result = []
    count = 0
    started = False  # to skip the very first '|'

    for ch in S:
        if ch == '|':
            if started:
                # we've just finished counting hyphens for one segment
                result.append(count)
                count = 0
            else:
                # this is the first '|' in the string
                started = True
        else:  # ch == '-'
            if started:
                count += 1

    # Output the reconstructed sequence
    print(" ".join(map(str, result)))

if __name__ == "__main__":
    main()