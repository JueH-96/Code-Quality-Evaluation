import bisect

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    n = int(data[0])
    m = int(data[1])
    a = list(map(int, data[2:2+n]))
    
    a.sort()
    
    candidates = set(a)
    for num in a:
        candidates.add(num - m)
    
    max_count = 0
    for x in candidates:
        left = bisect.bisect_left(a, x)
        right = bisect.bisect_left(a, x + m)
        max_count = max(max_count, right - left)
    
    print(max_count)

if __name__ == "__main__":
    main()