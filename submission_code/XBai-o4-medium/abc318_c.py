import sys

def main():
    n, d, p = map(int, sys.stdin.readline().split())
    f = list(map(int, sys.stdin.readline().split()))
    total = sum(f)
    sorted_f = sorted(f, reverse=True)
    
    prefix = [0]
    current = 0
    for val in sorted_f:
        current += val
        prefix.append(current)
    
    k_max = (n + d - 1) // d
    min_cost = float('inf')
    
    for k in range(k_max + 1):
        x = min(k * d, n)
        sum_x = prefix[x]
        cost = (total - sum_x) + k * p
        if cost < min_cost:
            min_cost = cost
    
    print(min_cost)

if __name__ == "__main__":
    main()