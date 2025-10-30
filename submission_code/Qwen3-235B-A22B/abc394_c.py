def main():
    import sys
    S = sys.stdin.read().strip()
    s = list(S)
    i = 0
    while i < len(s) - 1:
        if s[i] == 'W' and s[i+1] == 'A':
            # Replace with AC
            s[i] = 'A'
            s[i+1] = 'C'
            # Move back to check for possible new WA starting at i-1
            i = max(0, i - 1)
        else:
            i += 1
    print(''.join(s))

if __name__ == "__main__":
    main()