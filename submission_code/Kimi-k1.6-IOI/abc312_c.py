import bisect

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    n = int(data[0])
    m = int(data[1])
    A = list(map(int, data[2:2+n]))
    B = list(map(int, data[2+n:2+n+m]))
    
    A.sort()
    B.sort()
    
    low = 1
    high = 10**18  # Sufficiently large upper bound
    
    while low < high:
        mid = (low + high) // 2
        sellers = bisect.bisect_right(A, mid)
        buyers = m - bisect.bisect_left(B, mid)
        if sellers >= buyers:
            high = mid
        else:
            low = mid + 1
    
    print(low)

if __name__ == "__main__":
    main()