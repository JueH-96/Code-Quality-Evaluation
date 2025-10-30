import bisect

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    idx = 0
    N = int(data[idx])
    idx += 1
    
    X = list(map(int, data[idx:idx+N]))
    idx += N
    
    P = list(map(int, data[idx:idx+N]))
    idx += N
    
    # Compute prefix sum
    prefix = [0]
    for p in P:
        prefix.append(prefix[-1] + p)
    
    Q = int(data[idx])
    idx += 1
    
    results = []
    for _ in range(Q):
        L = int(data[idx])
        R = int(data[idx+1])
        idx += 2
        
        left = bisect.bisect_left(X, L)
        right = bisect.bisect_right(X, R) - 1
        if left > right:
            results.append(0)
        else:
            results.append(prefix[right+1] - prefix[left])
    
    print('
'.join(map(str, results)))

if __name__ == "__main__":
    main()