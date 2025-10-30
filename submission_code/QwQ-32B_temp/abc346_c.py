def main():
    import sys
    n, k = map(int, sys.stdin.readline().split())
    a = list(map(int, sys.stdin.readline().split()))
    
    total = k * (k + 1) // 2
    unique_a = set(a)
    sum_a = 0
    for num in unique_a:
        if num <= k:
            sum_a += num
    print(total - sum_a)

if __name__ == "__main__":
    main()