import bisect

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    n = int(data[0])
    m = int(data[1])
    p = int(data[2])
    A = list(map(int, data[3:3+n]))
    B = list(map(int, data[3+n:3+n+m]))
    
    B.sort()
    
    prefix = [0]
    current = 0
    for b in B:
        current += b
        prefix.append(current)
    
    sum_part = 0
    count_part = 0
    
    for a in A:
        x = p - a
        idx = bisect.bisect_right(B, x)
        sum_i = prefix[idx]
        cnt_i = idx
        sum_part += a * cnt_i + sum_i
        count_part += cnt_i
    
    total = sum_part + (n * m - count_part) * p
    print(total)

if __name__ == "__main__":
    main()