class Solution:
    def minimumPossibleSum(self, n: int, target: int) -> int:
        MOD = 10**9 + 7
        # Explanation:
        # We want to pick n distinct positive integers with minimum sum.
        # However, we cannot pick two numbers which sum to target.
        #
        # Notice that if we pick a number x with x < target,
        # its complementary number target - x (which is > x)
        # might appear if we blindly take the smallest numbers.
        # So, among the numbers less than target, they appear in complementary pairs:
        # (1, target-1), (2, target-2), ... and if target is even, then target/2 stands alone.
        # To avoid any dangerous pair, we can choose at most one element from each pair.
        #
        # It is optimal to choose the smallest numbers possible.
        # The best strategy is to choose as many from the lower segment [1, target-1] as possible,
        # but we cannot choose both numbers of any pair.
        # It turns out that we can safely pick the numbers 1, 2, ..., k where
        # k = (target - 1)//2.
        #
        # If n <= k, then we can simply use the first n natural numbers.
        # Otherwise, we take all numbers [1,...,k] and then we need to pick the remaining numbers from a region
        # where no forbidden pair exists. Notice that any number x >= target is safe because its complement
        # (target - x) would be <= 0.
        # Thus, we pick the next (n - k) numbers starting from target.
        
        k = (target - 1) // 2
        if n <= k:
            # sum of first n natural numbers
            return (n * (n + 1) // 2) % MOD
        else:
            # Sum of numbers 1,2,...,k
            sum_lower = k * (k + 1) // 2
            # We need to pick the remaining (n - k) numbers starting from target:
            m = n - k
            # The m numbers are: target, target+1, ..., target+(m-1)
            sum_higher = m * target + (m * (m - 1) // 2)
            return (sum_lower + sum_higher) % MOD

# Example usage and testing:
if __name__ == '__main__':
    # Provided examples:
    sol = Solution()
    print(sol.minimumPossibleSum(2, 3))  # Expected output: 4   Explanation: [1,3]
    print(sol.minimumPossibleSum(3, 3))  # Expected output: 8   Explanation: [1,3,4]
    print(sol.minimumPossibleSum(1, 1))  # Expected output: 1   Explanation: [1]
    
    # For standard input reading (if needed):
    import sys
    data = sys.stdin.read().strip().split()
    if data:
        n = int(data[0])
        target = int(data[1])
        print(sol.minimumPossibleSum(n, target))