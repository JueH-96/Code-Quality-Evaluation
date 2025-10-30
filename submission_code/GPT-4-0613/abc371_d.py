import sys
from bisect import bisect_left, bisect_right

def main():
    n = int(sys.stdin.readline())
    x = list(map(int, sys.stdin.readline().split()))
    p = list(map(int, sys.stdin.readline().split()))
    q = int(sys.stdin.readline())
    queries = [list(map(int, sys.stdin.readline().split())) for _ in range(q)]
    
    prefix_sum = [0]
    for i in range(n):
        prefix_sum.append(prefix_sum[-1] + p[i])
    
    for l, r in queries:
        left = bisect_left(x, l)
        right = bisect_right(x, r)
        print(prefix_sum[right] - prefix_sum[left])

if __name__ == "__main__":
    main()