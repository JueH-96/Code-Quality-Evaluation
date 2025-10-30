import bisect

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    n = int(data[0])
    m = int(data[1])
    d = int(data[2])
    
    a = list(map(int, data[3:3+n]))
    b = list(map(int, data[3+n:3+n+m]))
    
    b.sort()
    max_sum = -1
    
    for num in a:
        target_low = num - d
        target_high = num + d
        
        idx_low = bisect.bisect_left(b, target_low)
        idx_high = bisect.bisect_right(b, target_high) - 1
        
        if idx_low <= idx_high:
            current_b = b[idx_high]
            current_sum = num + current_b
            if current_sum > max_sum:
                max_sum = current_sum
    
    print(max_sum if max_sum != -1 else -1)

if __name__ == "__main__":
    main()