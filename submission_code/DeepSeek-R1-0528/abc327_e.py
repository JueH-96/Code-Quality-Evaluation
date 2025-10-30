import math

def main():
    n = int(input().strip())
    P = list(map(int, input().split()))
    
    denom = [0.0] * (n + 1)
    if n >= 1:
        power = 0.9
        denom[1] = 10.0 * (1.0 - power)
        for j in range(2, n + 1):
            power *= 0.9
            denom[j] = 10.0 * (1.0 - power)
    
    best_prev = [-10**18] * (n + 1)
    ans = -10**18
    
    for i in range(n):
        m = i + 1
        current = [0.0] * (m + 1)
        current[1] = P[i]
        
        for j in range(2, m + 1):
            current[j] = 0.9 * best_prev[j - 1] + P[i]
        
        for j in range(1, m + 1):
            if current[j] > best_prev[j]:
                best_prev[j] = current[j]
            rating_val = current[j] / denom[j] - 1200.0 / math.sqrt(j)
            if rating_val > ans:
                ans = rating_val
                
    print("{:.15f}".format(ans))

if __name__ == '__main__':
    main()