import sys

def main():
    data = sys.stdin.read().splitlines()
    if len(data) < 3:
        print("No")
        return
    try:
        K = int(data[0])
        S = data[1].strip()
        T = data[2].strip()
    except Exception:
        print("No")
        return
        
    n = len(S)
    m = len(T)
    
    if abs(n - m) > K:
        print("No")
        return
        
    INF = 10**9
    total_size = 2 * K + 1
    prev = [INF] * total_size
    
    for d in range(-K, K + 1):
        j = d
        if j < 0 or j > m:
            continue
        idx = d + K
        prev[idx] = j
        
    for i in range(1, n + 1):
        curr = [INF] * total_size
        for d in range(-K, K + 1):
            j = i + d
            if j < 0 or j > m:
                continue
            idx = d + K
            options = []
            
            if d + 1 <= K:
                prev_idx = (d + 1) + K
                if prev[prev_idx] != INF:
                    options.append(prev[prev_idx] + 1)
                    
            if d - 1 >= -K:
                curr_idx_prev = (d - 1) + K
                if curr[curr_idx_prev] != INF:
                    options.append(curr[curr_idx_prev] + 1)
                    
            if i > 0 and j > 0:
                prev_idx_same = d + K
                if prev[prev_idx_same] != INF:
                    if S[i - 1] == T[j - 1]:
                        options.append(prev[prev_idx_same])
                    else:
                        options.append(prev[prev_idx_same] + 1)
                        
            if options:
                curr[idx] = min(options)
                
        prev = curr
        
    d_final = m - n
    if d_final < -K or d_final > K:
        print("No")
        return
        
    idx_final = d_final + K
    if prev[idx_final] <= K:
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    main()