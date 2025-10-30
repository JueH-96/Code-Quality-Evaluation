def main():
    import sys
    input_str = sys.stdin.read().strip()
    if not input_str:
        return
    # The entire input is stored in input_str (there is one line: S)
    s = list(input_str)
    n = len(s)
    i = 0
    # Loop until we reach the second last character.
    while i < n - 1:
        if s[i] == 'W' and s[i+1] == 'A':
            # Replace "WA" with "AC"
            s[i] = 'A'
            s[i+1] = 'C'
            # When replaced, we may have created an occurrence earlier.
            # Move back one step if possible.
            i = max(i - 1, 0)
        else:
            i += 1
    # Convert list back to string and print result.
    sys.stdout.write("".join(s))

if __name__ == '__main__':
    main()