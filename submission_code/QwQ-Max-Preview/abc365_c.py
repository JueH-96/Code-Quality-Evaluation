import bisect

def main():
    import sys
    input = sys.stdin.read().split()
    n = int(input[0])
    m = int(input[1])
    a = list(map(int, input[2:2+n]))
    
    total = sum(a)
    if total <= m:
        print("infinite")
        return
    
    a.sort()
    prefix = [0]
    current = 0
    for num in a:
        current += num
        prefix.append(current)
    
    low = 0
    high = a[-1]
    best = 0
    
    while low <= high:
        mid = (low + high) // 2
        idx = bisect.bisect_right(a, mid)
        sum_less = prefix[idx]
        count = n - idx
        s = sum_less + mid * count
        
        if s <= m:
            best = mid
            low = mid + 1
        else:
            high = mid - 1
    
    print(best)

if __name__ == "__main__":
    main()