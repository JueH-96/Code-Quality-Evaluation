def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx += 1
    X = int(input[idx])
    idx += 1

    A = list(map(int, input[idx:idx+N]))
    idx += N
    B = list(map(int, input[idx:idx+N]))
    idx += N

    P = list(map(int, input[idx:idx+N]))
    idx += N
    Q = list(map(int, input[idx:idx+N]))
    idx += N

    C = []
    for j in range(N):
        if P[j] == X and Q[j] == X:
            C.append(j)
    
    if not C:
        print(-1)
        return
    
    required = set()
    for j in C:
        required.add(P[j])
        required.add(Q[j])
    
    for i in range(1, N+1):
        if i != X and i in required:
            print(-1)
            return
    
    print(len(C))

if __name__ == '__main__':
    main()