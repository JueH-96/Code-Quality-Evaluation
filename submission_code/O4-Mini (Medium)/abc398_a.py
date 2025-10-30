def main():
    import sys
    data = sys.stdin.read().strip()
    if not data:
        return
    N = int(data)
    # Initialize all characters to '-'
    s = ['-'] * N
    # If N is odd, place exactly one '=' in the middle
    if N % 2 == 1:
        s[N // 2] = '='
    else:
        # If N is even, place exactly two adjacent '=' in the middle
        mid = N // 2
        s[mid - 1] = s[mid] = '='
    # Output the resulting palindrome
    print(''.join(s))

if __name__ == "__main__":
    main()