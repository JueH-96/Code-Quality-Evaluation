def main():
    import sys

    data = sys.stdin.read().strip()
    if not data:
        return
    N = int(data)
    # Create a base string of all '-'
    s = ['-'] * N

    # For even N, place two '=' in the very middle (positions N/2 - 1 and N/2)
    # For odd N, place one '=' at the center (position N//2)
    if N % 2 == 0:
        mid = N // 2
        s[mid - 1] = '='
        s[mid] = '='
    else:
        s[N // 2] = '='

    # Output the result
    sys.stdout.write(''.join(s))


# Call main to execute the solution
if __name__ == "__main__":
    main()