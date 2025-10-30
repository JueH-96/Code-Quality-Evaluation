# YOUR CODE HERE
import sys
from bisect import bisect_right

def main():
    N, Q = map(int, input().split())
    R = list(map(int, input().split()))
    R.sort()

    prefix_sum = [0]
    for r in R:
        prefix_sum.append(prefix_sum[-1] + r)

    for _ in range(Q):
        X = int(input())
        index = bisect_right(prefix_sum, X) - 1
        print(index)

if __name__ == "__main__":
    main()