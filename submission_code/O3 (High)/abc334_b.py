import sys

def ceil_div(a: int, b: int) -> int:
    """
    Returns ⌈a / b⌉  for integers a and positive integer b.
    """
    return -(-a // b)

def main() -> None:
    data = sys.stdin.read().strip().split()
    if not data:
        return
    A, M, L, R = map(int, data)          # read input values

    # Compute smallest and largest k such that L ≤ A + kM ≤ R
    lower_k = ceil_div(L - A, M)          # k ≥ (L-A)/M  (ceiling)
    upper_k = (R - A) // M               # k ≤ (R-A)/M  (floor)

    count = upper_k - lower_k + 1
    if count < 0:
        count = 0
    print(count)

if __name__ == "__main__":
    main()