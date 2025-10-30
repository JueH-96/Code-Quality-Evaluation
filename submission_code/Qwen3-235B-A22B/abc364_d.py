import bisect
import sys

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    Q = int(input[ptr])
    ptr += 1
    
    a = list(map(int, input[ptr:ptr+N]))
    ptr += N
    a.sort()
    
    for _ in range(Q):
        b = int(input[ptr])
        k = int(input[ptr + 1])
        ptr += 2
        
        low = 0
        high = 2 * 10**8
        ans = high
        
        while low <= high:
            mid = (low + high) // 2
            left = b - mid
            right = b + mid
            
            # Find left count
            l = bisect.bisect_left(a, left)
            # Find right count
            r = bisect.bisect_right(a, right)
            cnt = r - l
            
            if cnt >= k:
                ans = mid
                high = mid - 1
            else:
                low = mid + 1
        
        print(ans)

if __name__ == "__main__":
    main()