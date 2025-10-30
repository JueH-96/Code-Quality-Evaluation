def main():
    import sys
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    X = int(input[ptr])
    ptr += 1
    Y = int(input[ptr])
    ptr += 1

    buses = []
    for _ in range(N-1):
        P = int(input[ptr])
        ptr += 1
        T = int(input[ptr])
        ptr += 1
        buses.append((P, T))
    
    Q = int(input[ptr])
    ptr += 1
    queries = []
    for _ in range(Q):
        queries.append(int(input[ptr]))
        ptr += 1
    
    for q in queries:
        t = q + X
        for p, ti in buses:
            t = ((t + p - 1) // p) * p + ti
        total = t + Y
        print(total)

if __name__ == '__main__':
    main()