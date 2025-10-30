def main():
    import sys

    # Read input
    N, D = map(int, sys.stdin.readline().split())
    S = sys.stdin.readline().strip()

    # Collect positions (0-indexed) of all cookies
    cookie_pos = [i for i, ch in enumerate(S) if ch == '@']

    # Positions of the D rightmost cookies to be eaten
    to_remove = set(cookie_pos[-D:])   # last D indices

    # Build the final string
    result = []
    for i, ch in enumerate(S):
        if i in to_remove:
            result.append('.')         # cookie has been eaten
        else:
            result.append(ch)          # unchanged (either '.' or remaining '@')
    print(''.join(result))

if __name__ == "__main__":
    main()