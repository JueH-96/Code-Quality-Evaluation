import sys

def main():
    import sys
    N, M = map(int, sys.stdin.readline().split())
    L = list(map(int, sys.stdin.readline().split()))
    
    max_L = max(L)
    sum_L_plus_spaces = sum(L) + (N - 1)
    
    low = max_L
    high = sum_L_plus_spaces
    ans = high
    
    while low <= high:
        mid = (low + high) // 2
        lines = 1
        current = 0
        ok = True
        
        for l in L:
            if current == 0:
                current = l
            else:
                if current + 1 + l <= mid:
                    current += 1 + l
                else:
                    lines += 1
                    current = l
                    if lines > M:
                        ok = False
                        break
        
        if ok:
            ans = mid
            high = mid - 1
        else:
            low = mid + 1
    
    print(ans)

if __name__ == '__main__':
    main()