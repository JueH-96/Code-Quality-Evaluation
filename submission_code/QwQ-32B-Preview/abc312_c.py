import sys
from bisect import bisect_right, bisect_left

def main():
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    A = list(map(int, data[2:2+N]))
    B = list(map(int, data[2+N:2+N+M]))
    
    A.sort()
    B.sort()
    
    def check(X):
        sellers = bisect_right(A, X)
        buyers = M - bisect_left(B, X)
        return sellers >= buyers
    
    low = 1
    high = 10**9
    while low < high:
        mid = (low + high) // 2
        if check(mid):
            high = mid
        else:
            low = mid + 1
    print(low)

if __name__ == '__main__':
    main()