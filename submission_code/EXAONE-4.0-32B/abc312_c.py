import sys
import bisect

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    
    n = int(data[0])
    m = int(data[1])
    A = list(map(int, data[2:2+n]))
    B = list(map(int, data[2+n:2+n+m]))
    
    min_A = min(A)
    max_B = max(B)
    
    if min_A > max_B:
        print(max_B + 1)
    else:
        A.sort()
        B.sort()
        low = min_A
        high = max_B
        ans = max_B + 1
        
        while low <= high:
            mid = (low + high) // 2
            sell_count = bisect.bisect_right(A, mid)
            idx = bisect.bisect_left(B, mid)
            buy_count = len(B) - idx
            
            if sell_count >= buy_count:
                ans = mid
                high = mid - 1
            else:
                low = mid + 1
                
        print(ans)

if __name__ == "__main__":
    main()