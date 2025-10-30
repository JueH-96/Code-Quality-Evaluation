def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    Q = int(data[1])
    S = data[2]
    queries = data[3:]
    
    # Preprocessing: find all positions where S[p] == S[p+1]
    consecutive = []
    for p in range(N-1):
        if S[p] == S[p+1]:
            consecutive.append(p+1)  # 1-based index
    
    # Process each query
    for i in range(Q):
        l = int(queries[2*i])
        r = int(queries[2*i+1])
        # Convert to 1-based indices for p
        # We need to count p where l <= p < r and S[p] == S[p+1]
        # Since consecutive list is 1-based, we need to find p in [l, r-1]
        low = l
        high = r - 1
        # Find the first p >= low
        left = 0
        right = len(consecutive) - 1
        first = -1
        while left <= right:
            mid = (left + right) // 2
            if consecutive[mid] >= low:
                first = mid
                right = mid - 1
            else:
                left = mid + 1
        # Find the last p <= high
        left2 = 0
        right2 = len(consecutive) - 1
        last = -1
        while left2 <= right2:
            mid = (left2 + right2) // 2
            if consecutive[mid] <= high:
                last = mid
                left2 = mid + 1
            else:
                right2 = mid - 1
        if first == -1 or last == -1:
            print(0)
        else:
            print(last - first + 1)

if __name__ == "__main__":
    main()