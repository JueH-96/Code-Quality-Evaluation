def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    P = []
    Q = []
    
    index = 1
    for _ in range(N):
        A = int(data[index])
        B = int(data[index+1])
        P.append((A, B))
        index += 2
    
    for _ in range(N):
        C = int(data[index])
        D = int(data[index+1])
        Q.append((C, D))
        index += 2
    
    # Sort P and Q based on x-coordinate
    P_sorted = sorted(P, key=lambda x: x[0])
    Q_sorted = sorted(Q, key=lambda x: x[0])
    
    # Assign Q_sorted to P_sorted in order
    R = []
    for i in range(N):
        R.append(Q_sorted.index(Q[i]) + 1)
    
    print(' '.join(map(str, R)))

if __name__ == "__main__":
    main()