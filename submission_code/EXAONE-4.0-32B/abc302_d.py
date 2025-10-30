import bisect
import sys

def main():
    data = sys.stdin.read().split()
    n = int(data[0])
    m = int(data[1])
    D = int(data[2])
    A = list(map(int, data[3:3+n]))
    B = list(map(int, data[3+n:3+n+m]))
    
    A.sort()
    B.sort()
    
    ans = -1
    for a in A:
        low_bound = a - D
        high_bound = a + D
        
        L = bisect.bisect_left(B, low_bound)
        R = bisect.bisect_right(B, high_bound) - 1
        
        if L <= R:
            total = a + B[R]
            if total > ans:
                ans = total
                
    print(ans)

if __name__ == "__main__":
    main()