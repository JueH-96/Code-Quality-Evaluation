def main():
    import sys
    # Read input from standard input
    tokens = sys.stdin.read().split()
    if not tokens:
        return

    # Parse the first three tokens: N (ignored after reading it), c1, and c2.
    N = int(tokens[0])
    c1 = tokens[1]
    c2 = tokens[2]
    
    # The remaining token is the string S.
    S = tokens[3]
    
    # Construct the answer by replacing each character that is not c1 with c2.
    result = "".join(ch if ch == c1 else c2 for ch in S)
    
    # Output the answer.
    print(result)

# Call main to execute the solution.
if __name__ == '__main__':
    main()