def main():
    import sys
    input = sys.stdin.read().split()
    n = int(input[0])
    k = list(map(int, input[1:n+1]))
    total = sum(k)
    min_max = float('inf')
    
    for mask in range(1 << n):
        subset_sum = 0
        for i in range(n):
            if mask & (1 << i):
                subset_sum += k[i]
        current_max = max(subset_sum, total - subset_sum)
        if current_max < min_max:
            min_max = current_max
    
    print(min_max)

if __name__ == "__main__":
    main()