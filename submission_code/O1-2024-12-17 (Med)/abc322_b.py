def main():
    import sys
    data = sys.stdin.read().strip().split()
    N, M = map(int, data[:2])
    S = data[2]
    T = data[3]

    is_prefix = (T[:N] == S)
    is_suffix = (T[-N:] == S)

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