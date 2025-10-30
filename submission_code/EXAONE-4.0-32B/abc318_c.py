import sys

def main():
    data = sys.stdin.read().split()
    n = int(data[0])
    d = int(data[1])
    p = int(data[2])
    fares = list(map(int, data[3:3+n]))
    
    total_fare = sum(fares)
    fares.sort(reverse=True)
    
    prefix = [0] * (n + 1)
    for i in range(1, n + 1):
        prefix[i] = prefix[i - 1] + fares[i - 1]
    
    k_max = (n + d - 1) // d
    ans = total_fare
    
    for k in range(1, k_max + 1):
        covered_days = min(k * d, n)
        cost_k = k * p + (total_fare - prefix[covered_days])
        if cost_k < ans:
            ans = cost_k
            
    print(ans)

if __name__ == "__main__":
    main()