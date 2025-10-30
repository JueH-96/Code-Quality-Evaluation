def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    idx = 0
    N = int(data[idx])
    idx += 1
    X = list(map(int, data[idx:idx+N]))
    idx += N
    Q = int(data[idx])
    idx += 1
    tasks = []
    for _ in range(Q):
        T = int(data[idx]) - 1  # convert to 0-based index
        G = int(data[idx + 1])
        tasks.append((T, G))
        idx += 2
    
    total = 0
    for T, G in tasks:
        current = X[T]
        total += abs(current - G)
        X[T] = G  # Update the position for future tasks
    
    print(total)

if __name__ == '__main__':
    main()