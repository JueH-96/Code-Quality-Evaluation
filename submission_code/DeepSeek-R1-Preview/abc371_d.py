import bisect

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    ptr = 0
    
    n = int(data[ptr])
    ptr += 1
    
    X = list(map(int, data[ptr:ptr + n]))
    ptr += n
    
    P = list(map(int, data[ptr:ptr + n]))
    ptr += n
    
    # Compute prefix sum
    prefix = [0] * (n + 1)
    for i in range(1, n + 1):
        prefix[i] = prefix[i - 1] + P[i - 1]
    
    q = int(data[ptr])
    ptr += 1
    
    results = []
    for _ in range(q):
        L = int(data[ptr])
        R = int(data[ptr + 1])
        ptr += 2
        
        left_idx = bisect.bisect_left(X, L)
        right_insert = bisect.bisect_right(X, R)
        right_idx = right_insert - 1
        
        if left_idx > right_idx:
            results.append(0)
        else:
            total = prefix[right_idx + 1] - prefix[left_idx]
            results.append(total)
    
    print('
'.join(map(str, results)))

if __name__ == '__main__':
    main()