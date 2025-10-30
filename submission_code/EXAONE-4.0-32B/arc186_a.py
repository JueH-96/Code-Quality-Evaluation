import sys

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    n = int(data[0])
    q = int(data[1])
    queries = list(map(int, data[2:2+q]))
    
    total = n * n
    dp = [[set() for _ in range(n+1)] for __ in range(n+1)]
    dp[0][0].add(0)
    
    for i in range(n + 1):
        for j in range(n + 1):
            for a in range(0, n - i + 1):
                for b in range(0, n - j + 1):
                    if a == 0 and b == 0:
                        continue
                    if a >= 1 and b >= 1:
                        if a < 2 or b < 2:
                            continue
                    current_vals = list(dp[i][j])
                    for val in current_vals:
                        new_val = val + a * b
                        if new_val > total:
                            continue
                        ni = i + a
                        nj = j + b
                        if ni <= n and nj <= n:
                            dp[ni][nj].add(new_val)
    
    results = []
    for k in queries:
        T_req = total - k
        if T_req in dp[n][n]:
            results.append("Yes")
        else:
            results.append("No")
            
    for res in results:
        print(res)

if __name__ == "__main__":
    main()