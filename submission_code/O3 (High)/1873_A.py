import sys

def can_be_abc(s: str) -> bool:
    """
    Returns True if string s (a permutation of 'a', 'b', 'c')
    can be turned into 'abc' using at most one swap.
    """
    if s == "abc":
        return True

    s_list = list(s)
    # Try every possible swap (there are only 3 for length 3)
    for i in range(3):
        for j in range(i + 1, 3):
            t = s_list.copy()
            t[i], t[j] = t[j], t[i]
            if "".join(t) == "abc":
                return True
    return False


def main() -> None:
    data = sys.stdin.read().split()        # tokenizes by all whitespace (robust to blank lines)
    t = int(data[0])
    idx = 1
    for _ in range(t):
        s = data[idx]
        idx += 1
        print("YES" if can_be_abc(s) else "NO")


if __name__ == "__main__":
    main()