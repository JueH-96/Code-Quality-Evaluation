import sys

def is_extended_abc(s: str) -> bool:
    """
    Returns True iff s can be written as A*B*C*, i.e. it is an Extended ABC string.
    """
    order = {'A': 0, 'B': 1, 'C': 2}
    prev_rank = -1                         # smaller than any real rank
    for ch in s:
        rank = order[ch]
        if rank < prev_rank:               # order violated (went backwards)
            return False
        prev_rank = rank
    return True

def main() -> None:
    s = sys.stdin.readline().strip()
    print("Yes" if is_extended_abc(s) else "No")

if __name__ == "__main__":
    main()