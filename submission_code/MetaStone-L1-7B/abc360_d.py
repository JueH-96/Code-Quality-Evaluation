import bisect

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    idx = 0
    N = int(data[idx])
    idx += 1
    T = int(data[idx])
    idx += 1
    
    S = data[idx]
    idx += 1
    X = list(map(int, data[idx:idx+N]))
    idx += N
    
    R = []
    L = []
    for i in range(N):
        if S[i] == '1':
            R.append(X[i])
        else:
            L.append(X[i])
    
    R.sort()
    total = 0
    delta = 2 * T + 0.2
    
    for x in L:
        low = x - delta
        high = x + delta
        
        left = bisect.bisect_right(R, low)
        right = bisect.bisect_left(R, high)
        
        total += right - left
    
    print(int(total))

if __name__ == "__main__":
    main()