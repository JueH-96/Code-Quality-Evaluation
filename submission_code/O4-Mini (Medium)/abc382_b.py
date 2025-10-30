def main():
    import sys
    input = sys.stdin.readline

    N, D = map(int, input().split())
    S = list(input().rstrip('
'))
    # Collect indices of cookies
    cookie_indices = [i for i, c in enumerate(S) if c == '@']
    # We will remove the D rightmost cookies
    to_remove = cookie_indices[-D:]
    for idx in to_remove:
        S[idx] = '.'
    # Output the final state
    print(''.join(S))

if __name__ == "__main__":
    main()