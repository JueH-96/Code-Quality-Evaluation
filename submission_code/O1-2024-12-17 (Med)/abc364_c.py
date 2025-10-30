def main():
    import sys
    data = sys.stdin.read().strip().split()
    
    N = int(data[0])
    X = int(data[1])
    Y = int(data[2])
    
    A = list(map(int, data[3:3+N]))
    B = list(map(int, data[3+N:3+2*N]))
    
    # Function to find the smallest k such that the sum of the top k elements > threshold
    # Returns None if no such k exists
    def smallest_k_exceed(seq, threshold):
        seq_sorted = sorted(seq, reverse=True)
        s = 0
        for i, val in enumerate(seq_sorted, 1):
            s += val
            if s > threshold:
                return i
        return None
    
    kA = smallest_k_exceed(A, X)  # fewest dishes needed to exceed X (by sweetness)
    kB = smallest_k_exceed(B, Y)  # fewest dishes needed to exceed Y (by saltiness)
    
    if kA is None and kB is None:
        # Cannot exceed X or Y in any arrangement
        print(N)
    elif kA is None:
        # Cannot exceed X, but can exceed Y
        print(kB)
    elif kB is None:
        # Cannot exceed Y, but can exceed X
        print(kA)
    else:
        # We can exceed either X or Y; choose the minimal k
        print(min(kA, kB))

# Do not forget to call main()
main()