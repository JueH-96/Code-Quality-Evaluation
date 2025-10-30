import bisect

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    A = list(map(int, data[2:2+N]))
    B = list(map(int, data[2+N:2+N+M]))
    
    A.sort()
    B.sort()
    
    low = 1
    high = 10**9 + 1
    
    while low < high:
        mid = (low + high) // 2
        sellers = bisect.bisect_right(A, mid)
        buyers = M - bisect.bisect_left(B, mid)
        if sellers >= buyers:
            high = mid
        else:
            low = mid + 1
    print(low)

if __name__ == '__main__':
    main()