import bisect

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    n = int(data[0])
    m = int(data[1])
    d = int(data[2])
    
    A = list(map(int, data[3:3+n]))
    B = list(map(int, data[3+n:3+n+m]))
    
    B.sort()
    
    max_sum = -1
    
    for a in A:
        low = a - d
        high = a + d
        idx = bisect.bisect_right(B, high) - 1
        if idx >= 0:
            b = B[idx]
            if b >= low:
                current_sum = a + b
                if current_sum > max_sum:
                    max_sum = current_sum
    
    print(max_sum)

if __name__ == "__main__":
    main()