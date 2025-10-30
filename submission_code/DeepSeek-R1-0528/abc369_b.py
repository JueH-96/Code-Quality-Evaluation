INF = 10**9

def main():
    import sys
    data = sys.stdin.read().splitlines()
    n = int(data[0])
    notes = []
    for i in range(1, n+1):
        parts = data[i].split()
        A = int(parts[0])
        S = parts[1]
        notes.append((A, S))
    
    dp_prev = [[INF] * 101 for _ in range(101)]
    
    for l in range(1, 101):
        for r in range(1, 101):
            dp_prev[l][r] = 0
            
    for A, S in notes:
        dp_curr = [[INF] * 101 for _ in range(101)]
        if S == 'L':
            for r_prev in range(1, 101):
                best_val = INF
                for l_prev in range(1, 101):
                    cost = abs(A - l_prev)
                    total = dp_prev[l_prev][r_prev] + cost
                    if total < best_val:
                        best_val = total
                if best_val < dp_curr[A][r_prev]:
                    dp_curr[A][r_prev] = best_val
        else:
            for l_prev in range(1, 101):
                best_val = INF
                for r_prev in range(1, 101):
                    cost = abs(A - r_prev)
                    total = dp_prev[l_prev][r_prev] + cost
                    if total < best_val:
                        best_val = total
                if best_val < dp_curr[l_prev][A]:
                    dp_curr[l_prev][A] = best_val
        dp_prev = dp_curr
        
    ans = INF
    for l in range(1, 101):
        for r in range(1, 101):
            if dp_prev[l][r] < ans:
                ans = dp_prev[l][r]
                
    print(ans)

if __name__ == "__main__":
    main()