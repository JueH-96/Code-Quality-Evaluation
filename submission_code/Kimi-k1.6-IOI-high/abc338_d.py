def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    M = int(data[1])
    X = list(map(int, data[2:2+M]))
    
    diff = [0] * (N + 2)  # Using N+2 to handle up to index N
    original_sum = 0
    
    for i in range(M-1):
        A = X[i]
        B = X[i+1]
        cw = (B - A) % N
        ccw = (A - B) % N
        d = min(cw, ccw)
        original_sum += d
        
        if cw < ccw:
            start = A - 1
            length = cw
        else:
            start = B - 1
            length = ccw
        
        end = (start + length - 1) % N
        value = N - 2 * d
        
        if start <= end:
            diff[start] += value
            if end + 1 < N:
                diff[end + 1] -= value
            else:
                diff[N] -= value
        else:
            diff[start] += value
            diff[N] -= value
            diff[0] += value
            if end + 1 < N:
                diff[end + 1] -= value
            else:
                diff[N] -= value
    
    current = 0
    min_sum = float('inf')
    for i in range(N):
        current += diff[i]
        if current < min_sum:
            min_sum = current
    print(original_sum + min_sum)

if __name__ == "__main__":
    main()