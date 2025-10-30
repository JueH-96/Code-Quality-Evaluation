import bisect

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    ptr = 0
    N = int(data[ptr])
    ptr += 1
    
    X = list(map(int, data[ptr:ptr+N]))
    ptr += N
    
    P = list(map(int, data[ptr:ptr+N]))
    ptr += N
    
    # Compute prefix sum
    prefix = [0]
    current = 0
    for p in P:
        current += p
        prefix.append(current)
    
    Q = int(data[ptr])
    ptr += 1
    
    results = []
    for _ in range(Q):
        L = int(data[ptr])
        R = int(data[ptr+1])
        ptr += 2
        
        left = bisect.bisect_left(X, L)
        right = bisect.bisect_right(X, R)
        ans = prefix[right] - prefix[left]
        results.append(ans)
    
    print('
'.join(map(str, results)))

if __name__ == '__main__':
    main()