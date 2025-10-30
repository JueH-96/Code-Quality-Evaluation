def main():
    n = int(input().strip())
    arr = list(map(int, input().split()))
    
    min_prefix = 0
    current = 0
    for num in arr:
        current += num
        if current < min_prefix:
            min_prefix = current
    
    total_sum = current
    M = max(0, -min_prefix)
    ans = M + total_sum
    print(ans)

if __name__ == '__main__':
    main()