import sys

def main():
    # Read first line: N, P, Q
    N_P_Q = sys.stdin.readline().strip().split()
    if not N_P_Q:
        return
    N, P, Q = map(int, N_P_Q)

    # Read dish prices
    D = list(map(int, sys.stdin.readline().strip().split()))
    # Ensure we have exactly N prices; if lines can be split across multiple lines, handle:
    while len(D) < N:
        D.extend(map(int, sys.stdin.readline().strip().split()))
    D = D[:N]  # trim extra if any

    # Minimum dish price
    min_dish = min(D)

    # Minimum total cost
    ans = min(P, Q + min_dish)

    print(ans)

if __name__ == "__main__":
    main()