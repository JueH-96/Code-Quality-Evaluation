from typing import List

class Solution:
    def maxDistinctElements(self, nums: List[int], k: int) -> int:
        # For each number, we can change it to any value in the interval [num-k, num+k].
        # We want to assign a number (within that interval) to as many elements as possible
        # while ensuring that all chosen numbers are distinct.
        #
        # A greedy approach works well here. For every element, define its interval [L, R] where:
        #   L = num - k, R = num + k.
        #
        # Now, we want to pick one integer per interval such that:
        #   - The chosen integer is within the interval.
        #   - It is different from every other chosen integer.
        #
        # The standard greedy strategy for such interval assignments is:
        #  1. Sort the intervals by their right endpoint R (and by L in case of tie).
        #  2. Initialize a variable 'current' to a very small number.
        #  3. For each interval [L, R] in sorted order, choose the smallest possible number
        #     that is not less than 'current' and falls inside the interval. If such a number is found,
        #     assign it and update 'current' to one more than that number (to ensure distinctness).
        #
        # This greedy algorithm maximizes the number of intervals for which we can assign a unique value.
        
        intervals = []
        for num in nums:
            L = num - k
            R = num + k
            intervals.append((L, R))
        
        # Sort intervals by their right boundary.
        intervals.sort(key=lambda x: x[1])
        
        current = -10**18  # A number smaller than any possible L (since k and nums[i] are non-negative)
        count = 0
        
        # Try to assign a unique number to as many intervals as possible.
        for L, R in intervals:
            # Choose the smallest candidate that is not less than current.
            candidate = current if current > L else L
            if candidate <= R:
                count += 1
                # Update current so that later assignments are distinct.
                current = candidate + 1
        
        return count

# The following solve() function is for local testing and handles input/output.
def solve():
    import sys
    data = sys.stdin.read().strip().split()
    
    # This input parsing attempts to support two potential input formats:
    # 1. The array is given in the form "[a,b,c,...]" followed by k.
    # 2. The first integer is the size of the array, then the array elements, then k.
    if not data:
        return
    if data[0].startswith('['):
        # Example: "[1,2,2,3,3,4]" for nums and then "2" for k.
        arr_str = data[0].strip('[]')
        if arr_str == "":
            nums = []
        else:
            nums = list(map(int, arr_str.split(',')))
        k = int(data[-1])
    else:
        # Example: "6 1 2 2 3 3 4 2", where the first token is n (number of elements).
        n = int(data[0])
        nums = list(map(int, data[1:1+n]))
        k = int(data[1+n])
        
    sol = Solution()
    ans = sol.maxDistinctElements(nums, k)
    sys.stdout.write(str(ans))

if __name__ == '__main__':
    solve()