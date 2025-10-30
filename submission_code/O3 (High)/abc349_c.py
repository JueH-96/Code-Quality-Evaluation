import sys

def is_subsequence(text: str, pattern: str) -> bool:
    """
    Return True if `pattern` is a subsequence of `text`.
    Both strings are expected to be in the same case.
    """
    i = 0                  # current position in pattern
    for ch in text:
        if ch == pattern[i]:
            i += 1
            if i == len(pattern):   # all characters matched
                return True
    return False            # ran out of text before finishing pattern


def main() -> None:
    S = sys.stdin.readline().strip()       # lowercase letters (3 … 1e5)
    T = sys.stdin.readline().strip()       # uppercase letters, length 3

    # 1. Try to match a length-3 subsequence from S
    if is_subsequence(S, T.lower()):
        print("Yes")
        return

    # 2. Try to match a length-2 subsequence from S, then append 'X'
    if T[2] == 'X' and is_subsequence(S, (T[0] + T[1]).lower()):
        print("Yes")
    else:
        print("No")


if __name__ == "__main__":
    main()