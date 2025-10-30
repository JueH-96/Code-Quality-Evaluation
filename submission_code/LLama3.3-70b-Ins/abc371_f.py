import sys

def read_ints():
    return list(map(int, sys.stdin.readline().split()))

def main():
    N = int(sys.stdin.readline())
    X = read_ints()
    Q = int(sys.stdin.readline())
    tasks = []
    for _ in range(Q):
        T, G = read_ints()
        tasks.append((T-1, G))

    total_movements = 0
    for T, G in tasks:
        # Move the person to the target position
        movements = abs(X[T] - G)
        total_movements += movements

        # Update the person's position
        X[T] = G

        # Move other persons if necessary
        for i in range(T-1, -1, -1):
            if X[i] > X[i+1]:
                movements = X[i] - X[i+1]
                total_movements += movements
                X[i] -= movements
        for i in range(T+1, N):
            if X[i] < X[i-1]:
                movements = X[i-1] - X[i]
                total_movements += movements
                X[i] += movements

    print(total_movements)

if __name__ == "__main__":
    main()