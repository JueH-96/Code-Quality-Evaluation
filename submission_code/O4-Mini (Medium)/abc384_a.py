def main():
    import sys

    data = sys.stdin.read().split()
    # First line: N, c1, c2
    # Second line: S
    # data = [N, c1, c2, S]
    # We don't actually need N since Python strings know their length
    _, c1, c2, S = data

    # Build the result by replacing every char not equal to c1 with c2
    result = ''.join(ch if ch == c1 else c2 for ch in S)

    # Output the result
    sys.stdout.write(result)

# Call main to execute the solution
main()