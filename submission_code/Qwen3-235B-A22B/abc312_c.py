import bisect

def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx += 1
    M = int(input[idx])
    idx += 1
    
    A = list(map(int, input[idx:idx+N]))
    idx += N
    B = list(map(int, input[idx:idx+M]))
    idx += M
    
    A.sort()
    B.sort()
    
    left = 0
    right = 10**18  # A large enough upper bound
    
    while left < right:
        mid = (left + right) // 2
        # Number of sellers that can sell at X=mid
        s = bisect.bisect_right(A, mid)
        # Number of buyers that can buy at X=mid
        pos = bisect.bisect_left(B, mid)
        t = M - pos
        if s >= t:
            right = mid
        else:
            left = mid + 1
    print(left)

if __name__ == "__main__":
    main()