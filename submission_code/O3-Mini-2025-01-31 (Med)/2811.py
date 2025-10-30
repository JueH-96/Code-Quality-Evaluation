class Solution:
    def minimumSum(self, n: int, k: int) -> int:
        # We want to construct an array of n distinct positive integers
        # such that no two distinct numbers in the array sum to k.
        # To achieve the minimum sum, we choose the smallest positive integers
        # possible, but we must avoid adding any number that would create a dangerous pair.
        #
        # We can use a "forbidden" set: whenever we add a number x (with x < k),
        # we mark its complement (k - x) as forbidden (unless x * 2 == k,
        # because that sum with itself doesn't count as the pair must be distinct).
        # Then, while iterating from 1 upward, if a candidate is in forbidden, we skip it.
        # This greedy strategy yields minimal sum.
        
        chosen = []
        forbidden = set()
        candidate = 1
        while len(chosen) < n:
            # If candidate is not forbidden (i.e. does not form a pair with an already chosen number)
            if candidate not in forbidden:
                chosen.append(candidate)
                # If candidate is less than k, then its complement k-candidate can no longer be chosen,
                # unless candidate == k-candidate (which means candidate*2 == k) because that doesn't matter
                # since the same element cannot be paired with itself.
                if candidate < k and candidate * 2 != k:
                    forbidden.add(k - candidate)
            candidate += 1
        
        return sum(chosen)

# Example usage:
if __name__ == "__main__":
    sol = Solution()
    print(sol.minimumSum(5, 4))  # Expected output: 18
    print(sol.minimumSum(2, 6))  # Expected output: 3