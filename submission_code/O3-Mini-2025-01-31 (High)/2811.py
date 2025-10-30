class Solution:
    def minimumSum(self, n: int, k: int) -> int:
        count = 0
        total = 0
        chosen = set()
        candidate = 1
        while count < n:
            # When candidate + some previously chosen number equals k,
            # that previously chosen number must be exactly (k - candidate).
            # But if candidate == k - candidate (i.e. candidate is k/2) then it is allowed
            # since we never pick the same number twice.
            if candidate != k - candidate and (k - candidate) in chosen:
                candidate += 1
                continue
            total += candidate
            chosen.add(candidate)
            count += 1
            candidate += 1
        return total

# For local testing:
if __name__ == "__main__":
    sol = Solution()
    # Example 1:
    n, k = 5, 4
    print(sol.minimumSum(n, k))  # Expected output: 18

    # Example 2:
    n, k = 2, 6
    print(sol.minimumSum(n, k))  # Expected output: 3