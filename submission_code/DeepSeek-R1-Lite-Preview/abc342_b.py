import sys

def main():
    tokens = sys.stdin.read().split()
    idx = 0
    N = int(tokens[idx]); idx += 1
    P = [int(tokens[idx + i]) for i in range(N)]; idx += N
    position_map = {P[i]: i + 1 for i in range(N)}
    Q = int(tokens[idx]); idx += 1
    for _ in range(Q):
        A = int(tokens[idx])
        B = int(tokens[idx + 1])
        idx += 2
        pos_A = position_map[A]
        pos_B = position_map[B]
        if pos_A < pos_B:
            print(A)
        else:
            print(B)

if __name__ == "__main__":
    main()