def main():
    import sys
    input = sys.stdin.readline

    N, D = map(int, input().split())
    S = list(input().rstrip())

    # Collect indices of all cookies '@'
    cookie_positions = [i for i, c in enumerate(S) if c == '@']
    # We will eat the rightmost D cookies, i.e., the last D in this list
    to_remove = cookie_positions[-D:]

    # Remove those cookies
    for idx in to_remove:
        S[idx] = '.'

    # Output the final state
    print(''.join(S))

if __name__ == "__main__":
    main()