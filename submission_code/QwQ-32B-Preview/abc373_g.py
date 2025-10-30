def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    idx = 1
    A = []
    B = []
    for _ in range(N):
        A.append(int(data[idx]))
        B.append(int(data[idx+1]))
        idx += 2
    C = []
    D = []
    for _ in range(N):
        C.append(int(data[idx]))
        D.append(int(data[idx+1]))
        idx += 2
    
    # Sort P points by x-coordinate and keep original indices
    p_sorted = sorted(range(N), key=lambda i: A[i])
    
    # Sort Q points by x-coordinate and keep original indices
    q_sorted = sorted(range(N), key=lambda i: C[i])
    
    # Create a mapping from original P index to its sorted position
    p_map = {p_sorted[i]: i for i in range(N)}
    
    # Determine R
    R = [q_sorted[p_map[i]] + 1 for i in range(N)]
    
    # Print the result
    print(' '.join(map(str, R)))

if __name__ == '__main__':
    main()