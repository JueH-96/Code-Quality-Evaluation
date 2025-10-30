import bisect

def main():
    import sys
    input = sys.stdin.read().split()
    n = int(input[0])
    m = int(input[1])
    a = list(map(int, input[2:2+n]))
    
    sum_a = sum(a)
    if sum_a <= m:
        print("infinite")
        return
    
    a.sort()
    prefix = [0] * (n + 1)
    for i in range(n):
        prefix[i+1] = prefix[i] + a[i]
    
    max_x = a[-1]
    low = 0
    high = max_x
    best = 0
    
    while low <= high:
        mid = (low + high) // 2
        k = bisect.bisect_right(a, mid)
        total = prefix[k] + mid * (n - k)
        if total <= m:
            best = mid
            low = mid + 1
        else:
            high = mid - 1
    
    print(best)

if __name__ == "__main__":
    main()