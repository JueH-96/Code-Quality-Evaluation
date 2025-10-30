class Solution:
    def minimumSum(self, n: int, k: int) -> int:
        # We want a k-avoiding array with minimal sum.
        # The greedy strategy is to pick the smallest positive integers,
        # but skip any number if its complement (k - candidate) is already chosen.
        # Since the array elements are distinct, we are safe to choose a number which is exactly k/2 when k is even,
        # because it won't be paired with another identical element.
        
        chosen = set()
        total = 0
        candidate = 1
        
        while len(chosen) < n:
            # If the complement (k - candidate) has already been chosen, then choosing candidate
            # would form a forbidden pair (candidate + (k - candidate) = k). So skip candidate.
            if (k - candidate) in chosen:
                candidate += 1
                continue
            
            # Otherwise, we can choose candidate.
            chosen.add(candidate)
            total += candidate
            candidate += 1
        
        return total


# Example usage and simple testing:
if __name__ == "__main__":
    sol = Solution()
    # Example 1: n = 5, k = 4 -> Expected output: 18
    print(sol.minimumSum(5, 4))
    # Example 2: n = 2, k = 6 -> Expected output: 3
    print(sol.minimumSum(2, 6))