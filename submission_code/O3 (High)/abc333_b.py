def main() -> None:
    import sys

    # vertices in the order they appear around the pentagon
    order = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}

    s = sys.stdin.readline().strip()   # S1S2
    t = sys.stdin.readline().strip()   # T1T2

    def key(pair: str) -> int:
        """Return 1 for a side, 2 for a diagonal."""
        a, b = order[pair[0]], order[pair[1]]
        diff = abs(a - b)
        return min(diff, 5 - diff)     # 1 or 2

    print('Yes' if key(s) == key(t) else 'No')


if __name__ == '__main__':
    main()