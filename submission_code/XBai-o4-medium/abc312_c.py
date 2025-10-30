import bisect

def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    n = int(input[idx])
    idx += 1
    m = int(input[idx])
    idx += 1
    
    A = list(map(int, input[idx:idx+n]))
    idx += n
    B = list(map(int, input[idx:idx+m]))
    idx += m
    
    A.sort()
    B.sort()
    
    low = 1
    high = B[-1] + 1
    
    while low < high:
        mid = (low + high) // 2
        sa = bisect.bisect_right(A, mid)
        sb = m - bisect.bisect_left(B, mid)
        if sa >= sb:
            high = mid
        else:
            low = mid + 1
    print(low)
    
if __name__ == '__main__':
    main()