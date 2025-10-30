def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    idx = 0
    N = int(data[idx])
    M = int(data[idx+1])
    idx +=2
    A = list(map(int, data[idx:idx+N]))
    idx += N
    B = list(map(int, data[idx:idx+M]))
    
    if not A:
        for _ in range(M):
            print(-1)
        return
    
    # Compute prefix min
    prefix_min = [A[0]]
    for i in range(1, N):
        prefix_min.append(min(prefix_min[-1], A[i]))
    
    # Process each B_j
    results = []
    for Bj in B:
        left = 0
        right = N -1
        ans = -1
        while left <= right:
            mid = (left + right) //2
            if prefix_min[mid] <= Bj:
                ans = mid
                right = mid -1
            else:
                left = mid +1
        results.append(ans +1 if ans !=-1 else -1)
    
    print('
'.join(map(str, results)))

if __name__ == "__main__":
    main()