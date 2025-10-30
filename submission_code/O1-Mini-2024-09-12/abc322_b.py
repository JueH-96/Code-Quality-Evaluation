# YOUR CODE HERE
def main():
    import sys
    lines = sys.stdin.read().splitlines()
    if not lines:
        print(3)
        return
    first_line = lines[0].strip()
    if not first_line:
        print(3)
        return
    N, M = map(int, first_line.split())
    if len(lines) < 3:
        print(3)
        return
    S = lines[1].strip()
    T = lines[2].strip()
    is_prefix = T.startswith(S)
    is_suffix = T.endswith(S)
    if is_prefix and is_suffix:
        print(0)
    elif is_prefix:
        print(1)
    elif is_suffix:
        print(2)
    else:
        print(3)

if __name__ == "__main__":
    main()