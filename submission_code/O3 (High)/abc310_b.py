import sys

def main() -> None:
    data = sys.stdin.read().strip().split()
    if not data:
        return
    it = iter(data)
    N = int(next(it))
    M = int(next(it))          # M is unused except for reading, but kept for clarity.

    prices = []
    funcs  = []                # list[set[int]]

    for _ in range(N):
        p   = int(next(it))
        c   = int(next(it))
        fns = {int(next(it)) for _ in range(c)}
        prices.append(p)
        funcs.append(fns)

    # Check every ordered pair (i, j)
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            # 1. P_i >= P_j
            if prices[i] < prices[j]:
                continue

            # 2. functions_i ⊆ functions_j
            if not funcs[i].issubset(funcs[j]):
                continue

            # 3. (P_i > P_j) OR (functions_j strictly superset functions_i)
            if prices[i] > prices[j] or len(funcs[j]) > len(funcs[i]):
                print("Yes")
                return

    print("No")

if __name__ == "__main__":
    main()