def main():
    import sys
    input = sys.stdin.readline

    N = int(input().strip())
    # Start with all '-' characters
    s = ['-'] * N

    # If N is odd, place a single '=' in the middle.
    # If N is even, place two adjacent '=' in the two central positions.
    if N % 2 == 1:
        s[N // 2] = '='
    else:
        s[N // 2 - 1] = '='
        s[N // 2] = '='

    # Output the resulting palindrome
    print(''.join(s))


# Call main to execute
main()