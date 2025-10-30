import sys

def main():
    n, k, x = map(int, sys.stdin.readline().split())
    T = list(map(int, sys.stdin.readline().split()))
    
    prefix = [0] * (n + 1)
    for i in range(n):
        prefix[i+1] = prefix[i] + T[i]
    
    INF = float('inf')
    dp_dis = [INF] * (n + 1)
    dp_next = [0] * (n + 1)
    
    dp_dis[0] = 0
    dp_next[0] = 0
    
    for i in range(1, n + 1):
        best_dis = INF
        best_next = 0
        for m in range(1, k + 1):
            j = i - m
            if j < 0:
                continue
            prev_dis = dp_dis[j]
            prev_next = dp_next[j]
            current_T = T[i-1]
            s = max(prev_next, current_T)
            sum_T = prefix[i] - prefix[j]
            added = m * s - sum_T
            new_dis = prev_dis + added
            new_next = s + x
            
            if new_dis < best_dis:
                best_dis = new_dis
                best_next = new_next
            elif new_dis == best_dis:
                if new_next < best_next:
                    best_next = new_next
        dp_dis[i] = best_dis
        dp_next[i] = best_next
    
    print(dp_dis[n])

if __name__ == "__main__":
    main()