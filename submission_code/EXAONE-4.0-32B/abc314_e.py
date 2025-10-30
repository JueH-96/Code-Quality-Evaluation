import sys

def main():
    data = sys.stdin.read().split()
    if not data:
        print(0.0)
        return
    
    n = int(data[0])
    M_val = int(data[1])
    wheels = []
    index = 2
    for _ in range(n):
        c = int(data[index])
        index += 1
        p = int(data[index])
        index += 1
        outcomes = list(map(int, data[index:index+p]))
        index += p
        wheels.append((c, p, outcomes))
    
    if M_val == 0:
        print(0.0)
        return
        
    dp = [0.0] * M_val
    
    for x in range(M_val-1, -1, -1):
        best = float('inf')
        for wheel in wheels:
            C, P, outcomes = wheel
            total_nonzero = 0.0
            count_zero = 0
            for s in outcomes:
                ns = x + s
                if ns >= M_val:
                    continue
                if s == 0:
                    count_zero += 1
                else:
                    total_nonzero += dp[ns]
            denom = 1.0 - (count_zero / P)
            candidate = (C + total_nonzero / P) / denom
            if candidate < best:
                best = candidate
        dp[x] = best
    
    print(dp[0])

if __name__ == "__main__":
    main()