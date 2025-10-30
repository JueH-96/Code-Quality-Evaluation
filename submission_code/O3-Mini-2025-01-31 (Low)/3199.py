class Solution:
    def distributeCandies(self, n: int, limit: int) -> int:
        count = 0
        for a in range(limit + 1):
            for b in range(limit + 1):
                c = n - a - b
                if 0 <= c <= limit:
                    count += 1
        return count

# Example usage:
if __name__ == "__main__":
    sol = Solution()
    print(sol.distributeCandies(5, 2))  # Expected output: 3
    print(sol.distributeCandies(3, 3))  # Expected output: 10