from typing import List

class Solution:
    def countPairs(self, nums: List[int]) -> int:
        # Helper function to compute the set of numbers reachable from n
        # by swapping any two digits exactly once.
        def compute_swaps(n: int) -> set:
            s = list(str(n))
            res = set()
            if len(s) < 2:
                return res
            for i in range(len(s)):
                for j in range(i+1, len(s)):
                    s_copy = s[:]  # copy the list of digits
                    s_copy[i], s_copy[j] = s_copy[j], s_copy[i]
                    res.add(int("".join(s_copy)))  # converting to int allows leading zeros to drop
            return res

        # Precompute transformation sets for each distinct number in nums.
        transforms = {}
        for num in set(nums):
            transforms[num] = compute_swaps(num)
        
        count = 0
        n = len(nums)
        for i in range(n):
            for j in range(i+1, n):
                a, b = nums[i], nums[j]
                # If already equal, no swap needed.
                if a == b:  
                    count += 1
                else:
                    # Check if one swap in a yields b, or one swap in b yields a.
                    if (b in transforms[a]) or (a in transforms[b]):
                        count += 1
        return count

# Example usage:
if __name__ == "__main__":
    sol = Solution()
    print(sol.countPairs([3,12,30,17,21]))  # Expected output: 2
    print(sol.countPairs([1,1,1,1,1]))       # Expected output: 10
    print(sol.countPairs([123,231]))         # Expected output: 0