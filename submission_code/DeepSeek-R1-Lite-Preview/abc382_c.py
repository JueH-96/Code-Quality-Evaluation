import bisect

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    N, M = int(data[0]), int(data[1])
    A = list(map(int, data[2:2+N]))
    B = list(map(int, data[2+N:2+N+M]))
    
    # Create list of (A_i, i) and sort by A_i
    sorted_A = sorted([(A[i], i+1) for i in range(N)], key=lambda x: x[0])
    
    # Compute prefix_min
    prefix_min = [0] * N
    prefix_min[0] = sorted_A[0][1]
    for i in range(1, N):
        prefix_min[i] = min(prefix_min[i-1], sorted_A[i][1])
    
    # Prepare a list of A_i sorted for binary search
    A_sorted = [x[0] for x in sorted_A]
    
    # Process each B_j
    results = []
    for b in B:
        index = bisect.bisect_right(A_sorted, b)
        if index == 0:
            results.append(-1)
        else:
            results.append(prefix_min[index-1])
    
    # Print all results
    print('
'.join(map(str, results)))

if __name__ == '__main__':
    main()