import sys

def main():
    import sys
    input = sys.stdin.read().split()
    K = int(input[0])
    S = input[1]
    T = input[2]
    n = len(S)
    m = len(T)
    
    if abs(n - m) > K:
        print("No")
        return
    
    INF = float('inf')
    prev = [INF] * (m + 2)
    for j in range(min(m, K) + 1):
        prev[j] = j
    
    for i in range(1, n + 1):
        curr = [INF] * (m + 2)
        # Handle j=0
        if i <= K:
            curr[0] = i
        else:
            curr[0] = INF
        
        window_start = max(1, i - K)
        window_end = min(m, i + K)
        
        for j in range(window_start, window_end + 1):
            cost = 0 if S[i-1] == T[j-1] else 1
            replace = prev[j-1] + cost
            insert = curr[j-1] + 1
            delete = prev[j] + 1
            curr[j] = min(replace, insert, delete)
        
        prev = curr
    
    if prev[m] <= K:
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    main()