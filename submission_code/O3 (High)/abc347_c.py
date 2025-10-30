import sys

def main() -> None:
    # Read first line
    N, A, B = map(int, sys.stdin.readline().split())
    L = A + B                          # length of one week

    # Read all D_i (they may span several lines)
    D = list(map(int, sys.stdin.read().split()))
    # Keep only remainders modulo L and remove duplicates
    mods = sorted(set(d % L for d in D))

    # Build an extended list to handle the wrap-around gap
    mods.append(mods[0] + L)           # append first element shifted by one week

    # Find the longest sequence of consecutive days with no plan
    max_empty = 0
    for i in range(len(mods) - 1):
        gap = mods[i + 1] - mods[i] - 1   # days strictly between two plans
        if gap > max_empty:
            max_empty = gap

    # If we have at least B consecutive free days, all plans can be on holidays
    print("Yes" if max_empty >= B else "No")

if __name__ == "__main__":
    main()