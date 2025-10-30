class Solution:
    def numberOfChild(self, n: int, k: int) -> int:
        # The movement is cyclic with period = 2*(n-1)
        period = 2 * (n - 1)
        # compute effective steps inside one cycle
        r = k % period
        # If r is within the forward part, simply the child index is r.
        # Otherwise, we are in the backward part, so the index is period - r.
        if r <= n - 1:
            return r
        else:
            return period - r

# Example usage:
if __name__ == "__main__":
    sol = Solution()
    # Test examples
    print(sol.numberOfChild(3, 5))  # Output: 1
    print(sol.numberOfChild(5, 6))  # Output: 2
    print(sol.numberOfChild(4, 2))  # Output: 2