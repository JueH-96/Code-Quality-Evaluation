import sys

def main():
    data = sys.stdin.read().split()
    ptr = 0
    N = int(data[ptr])
    ptr += 1
    P = []
    for _ in range(N):
        P.append(int(data[ptr]))
        ptr += 1
    positions = [0] * (N + 1)
    for i in range(1, N + 1):
        positions[P[i - 1]] = i
    Q = int(data[ptr])
    ptr += 1
    for _ in range(Q):
        A = int(data[ptr])
        ptr += 1
        B = int(data[ptr])
        ptr += 1
        if positions[A] < positions[B]:
            print(A)
        else:
            print(B)

if __name__ == '__main__':
    main()