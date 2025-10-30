import bisect

def main():
    import sys
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    M = int(input[ptr])
    ptr += 1
    
    A = list(map(int, input[ptr:ptr+N]))
    ptr += N
    B = list(map(int, input[ptr:ptr+M]))
    ptr += M
    
    # Create list of (A_i, original index) sorted by A_i
    sorted_pairs = sorted( ( (A[i], i+1) for i in range(N) ), key=lambda x: x[0] )
    sorted_A = [a for a, _ in sorted_pairs]
    
    # Precompute prefix_min
    prefix_min = []
    current_min = float('inf')
    for a, idx in sorted_pairs:
        if idx < current_min:
            current_min = idx
        prefix_min.append(current_min)
    
    # Process each B_j
    for b in B:
        pos = bisect.bisect_right(sorted_A, b) - 1
        if pos < 0:
            print(-1)
        else:
            print(prefix_min[pos])
    
if __name__ == "__main__":
    main()