def main():
    import sys
    s = sys.stdin.readline().strip()
    res = []
    for ch in s:
        res.append(ch)
        # If the last three characters form "ABC", remove them
        if len(res) >= 3 and res[-3] == 'A' and res[-2] == 'B' and res[-1] == 'C':
            res.pop()
            res.pop()
            res.pop()
    # Output the remaining characters
    sys.stdout.write(''.join(res))

if __name__ == '__main__':
    main()