# YOUR CODE HERE
import sys
from functools import lru_cache

MOD = 998244353

def main():
    N = int(sys.stdin.readline())
    balls = []
    for _ in range(N):
        x, y = map(int, sys.stdin.readline().split())
        balls.append((x, y))
    
    # Sort balls based on x-coordinate
    balls.sort()
    
    # Extract y-coordinates in order of sorted x
    y_list = [y for x, y in balls]
    
    # Precompute the longest increasing subsequence (LIS) for y_list
    # We will use a dynamic programming approach to find the length of the LIS
    # and then use it to determine the number of possible sets.
    
    # To find the number of possible sets, we need to find the number of ways to choose
    # a subset of balls such that no two balls are in the same LIS.
    # This is equivalent to finding the number of antichains in the poset defined by the
    # dominance relation.
    
    # However, a simpler approach is to realize that the number of possible sets is equal
    # to the number of ways to choose a subset of the LIS, which is 2^k, where k is the length
    # of the LIS. But this is not directly applicable.
    
    # Instead, we can use the fact that the number of possible sets is equal to the number of
    # ways to choose a subset of the balls such that no two balls are in the same dominance
    # relation. This is equivalent to the number of ways to choose a subset of the LIS.
    
    # To find the number of possible sets, we can use the following approach:
    # 1. Find the length of the longest increasing subsequence (LIS) in the y_list.
    # 2. The number of possible sets is 2^k, where k is the length of the LIS.
    
    # Let's implement the LIS algorithm.
    
    dp = []
    for y in y_list:
        low, high = 0, len(dp) - 1
        while low <= high:
            mid = (low + high) // 2
            if dp[mid] < y:
                low = mid + 1
            else:
                high = mid - 1
        if low < len(dp):
            dp[low] = y
        else:
            dp.append(y)
    
    k = len(dp)
    
    # The number of possible sets is 2^k
    result = pow(2, k, MOD)
    
    print(result)

if __name__ == "__main__":
    main()