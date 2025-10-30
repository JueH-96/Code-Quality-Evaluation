import bisect

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    idx = 0
    N = int(data[idx])
    M = int(data[idx+1])
    idx +=2
    A = list(map(int, data[idx:idx+N]))
    idx +=N
    B = list(map(int, data[idx:idx+M]))
    
    # Sort people by their gourmet level, keeping track of original indices (1-based)
    sorted_A = sorted(zip(A, range(1, N+1)))
    a_values = [x[0] for x in sorted_A]
    
    # Precompute the minimum index up to each position in the sorted list
    min_index = [0] * len(sorted_A)
    min_index[0] = sorted_A[0][1]
    for i in range(1, len(sorted_A)):
        min_index[i] = min(min_index[i-1], sorted_A[i][1])
    
    # Process each sushi
    result = []
    for bj in B:
        k = bisect.bisect_right(a_values, bj) -1
        if k >=0:
            result.append(str(min_index[k]))
        else:
            result.append("-1")
    
    print('
'.join(result))

if __name__ == "__main__":
    main()