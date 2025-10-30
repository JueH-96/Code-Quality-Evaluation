import sys

def main() -> None:
    # Read N, S, K
    N_S_K = sys.stdin.readline().strip().split()
    while len(N_S_K) < 3:
        N_S_K += sys.stdin.readline().strip().split()
    N, S, K = map(int, N_S_K)

    total = 0
    for _ in range(N):
        # Read P_i, Q_i
        while True:
            line = sys.stdin.readline()
            if line.strip():
                break
        P, Q = map(int, line.strip().split())
        total += P * Q

    # Determine shipping fee
    fee = 0 if total >= S else K

    # Output final amount
    print(total + fee)

if __name__ == "__main__":
    main()