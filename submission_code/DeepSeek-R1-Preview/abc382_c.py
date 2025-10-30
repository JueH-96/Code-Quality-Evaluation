import bisect

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    idx = 0
    N = int(data[idx])
    idx += 1
    M = int(data[idx])
    idx += 1
    
    A = list(map(int, data[idx:idx+N]))
    idx += N
    B = list(map(int, data[idx:idx+M]))
    idx += M
    
    # Create sorted_A sorted by A_i and then index
    sorted_A = sorted((A[i], i + 1) for i in range(N))
    A_values = [a for a, i in sorted_A]
    pre_min = []
    if not sorted_A:
        # Handle empty case, though N >=1 per constraints
        pass
    else:
        pre_min = [0] * len(sorted_A)
        pre_min[0] = sorted_A[0][1]
        for k in range(1, len(sorted_A)):
            pre_min[k] = min(pre_min[k-1], sorted_A[k][1])
    
    for b in B:
        # Find the largest index k where A_values[k] <= b
        k = bisect.bisect_right(A_values, b) - 1
        if k < 0:
            print(-1)
        else:
            print(pre_min[k])

if __name__ == '__main__':
    main()