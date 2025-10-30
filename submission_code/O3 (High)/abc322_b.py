import sys

def main() -> None:
    # Read input
    data = sys.stdin.read().strip().splitlines()
    if not data:
        return
    N, M = map(int, data[0].split())
    S = data[1].rstrip('
')
    T = data[2].rstrip('
')

    # Check prefix and suffix conditions
    is_prefix = T[:N] == S
    is_suffix = T[-N:] == S

    # Determine and print result
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