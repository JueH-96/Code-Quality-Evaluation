import bisect

def find_min_x():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    M = int(input[1])
    A = list(map(int, input[2:2+N]))
    B = list(map(int, input[2+N:2+N+M]))
    
    A.sort()
    B.sort()
    
    low = 1
    high = 10**14  # A safe upper bound
    
    while low < high:
        mid = (low + high) // 2
        # Number of sellers: A_i <= mid
        sellers = bisect.bisect_right(A, mid)
        # Number of buyers: B_i >= mid
        buyers = M - bisect.bisect_left(B, mid)
        if sellers >= buyers:
            high = mid
        else:
            low = mid + 1
    return low

print(find_min_x())