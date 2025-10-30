import bisect

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    ptr = 0
    N = int(data[ptr])
    ptr += 1
    X = list(map(int, data[ptr:ptr+N]))
    ptr += N
    Q = int(data[ptr])
    ptr += 1
    tasks = []
    for _ in range(Q):
        T = int(data[ptr])-1  # 0-based index
        G = int(data[ptr+1])
        tasks.append((T, G))
        ptr += 2
    
    total = 0
    for T, G in tasks:
        x = X[T]
        if G < x:
            left = bisect.bisect_left(X, G)
            right = bisect.bisect_right(X, G)
            K = left - (T + 1)
        else:
            left = bisect.bisect_left(X, G)
            right = bisect.bisect_right(X, G)
            K = (right - 1) - T
        steps = abs(G - x) + 2 * K
        total += steps
        X[T] = G  # Update the position after moving
    
    print(total)

if __name__ == '__main__':
    main()