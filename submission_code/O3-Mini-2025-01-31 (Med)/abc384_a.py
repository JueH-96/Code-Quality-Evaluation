def main():
    import sys
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    # First element is N, then c1, then c2, then string S in next token.
    N = int(input_data[0])
    c1 = input_data[1]
    c2 = input_data[2]
    S = input_data[3]
    
    # Build the output string by replacing each character of S that is not c1 with c2.
    result = ''.join([ch if ch == c1 else c2 for ch in S])
    print(result)

if __name__ == '__main__':
    main()