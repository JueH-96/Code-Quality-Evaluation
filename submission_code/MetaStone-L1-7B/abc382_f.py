import bisect

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    idx = 0
    N = int(data[idx])
    idx += 1
    H = int(data[idx])
    idx += 1
    W = int(data[idx])
    idx += 1
    
    bars = []
    for _ in range(N):
        R = int(data[idx])
        idx += 1
        L = int(data[idx])
        idx += 1
        C = int(data[idx])
        idx += 1
        bars.append((R, C, L))
    
    bars_sorted = sorted(bars, key=lambda x: x[1])
    result = [0] * N
    
    for i in range(N-1, -1, -1):
        R_i, C_i, L_i = bars_sorted[i]
        K = 0
        
        for j in range(i+1, N):
            R_j, C_j, L_j = bars_sorted[j]
            if C_j < C_i and R_j < R_i:
                K += 1
        
        result[i] = R_i + K
    
    for res in result:
        print(res)

if __name__ == '__main__':
    main()