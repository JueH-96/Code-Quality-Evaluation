def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    index = 1
    
    P = []
    Q = []
    
    for i in range(N):
        A = int(data[index])
        B = int(data[index + 1])
        P.append((A, B, i + 1))  # (x, y, original index)
        index += 2
    
    for i in range(N):
        C = int(data[index])
        D = int(data[index + 1])
        Q.append((C, D, i + 1))  # (x, y, original index)
        index += 2
    
    # Sort P and Q by x-coordinate
    P_sorted = sorted(P, key=lambda x: (x[0], x[1]))
    Q_sorted = sorted(Q, key=lambda x: (x[0], x[1]))
    
    # Create a mapping from sorted Q to original index
    Q_original_indices = [q[2] for q in Q_sorted]
    
    # Permutation R: for each i, R_i is the original index of Q_sorted[i]
    R = Q_original_indices
    
    # Output the permutation
    print(' '.join(map(str, R)))

if __name__ == '__main__':
    main()