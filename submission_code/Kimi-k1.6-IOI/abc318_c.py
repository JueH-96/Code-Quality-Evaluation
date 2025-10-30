def main():
    import sys
    input = sys.stdin.read().split()
    n = int(input[0])
    d = int(input[1])
    p = int(input[2])
    f = list(map(int, input[3:3+n]))
    
    f_sorted = sorted(f, reverse=True)
    prefix = [0]
    for num in f_sorted:
        prefix.append(prefix[-1] + num)
    total = prefix[-1]
    
    k_max = (n + d - 1) // d
    min_cost = float('inf')
    
    for k in range(k_max + 1):
        cover = min(k * d, n)
        cost = k * p + (total - prefix[cover])
        if cost < min_cost:
            min_cost = cost
    
    print(min_cost)

if __name__ == "__main__":
    main()