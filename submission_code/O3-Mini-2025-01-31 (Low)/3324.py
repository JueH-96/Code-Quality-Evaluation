from typing import List
from collections import Counter

class Solution:
    def isPossibleToSplit(self, nums: List[int]) -> bool:
        n = len(nums)
        m = n // 2  # each part must have m elements
        
        # Count frequency of each number.
        cnt = Counter(nums)
        
        # If any element appears more than twice, we cannot split without duplicating in one half.
        for freq in cnt.values():
            if freq > 2:
                return False
        
        # Let f be the count of elements that appear exactly twice.
        # Those elements must appear once in each half.
        f = sum(1 for v in cnt.values() if v == 2)
        # Let s be the count of elements that appear exactly once.
        s = sum(1 for v in cnt.values() if v == 1)
        
        # Each half already gets f distinct elements from the pairs.
        # So, each half needs m - f more distinct elements.
        # All such additional numbers must come from singles, and since singles
        # can be used only in one half, we need at least 2 * (m - f) singles overall.
        return s >= 2 * (m - f)


# The following solve() function is provided to allow this solution to be run
# as a standalone script that reads from standard input.
def solve():
    import sys
    data = sys.stdin.read().strip().split()
    if not data:
        return
    # Convert input tokens into integers.
    nums = list(map(int, data))
    
    solution = Solution()
    result = solution.isPossibleToSplit(nums)
    sys.stdout.write("true" if result else "false")


# When running the module directly, call solve().
if __name__ == '__main__':
    solve()