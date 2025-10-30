from typing import List

class Solution:
    def countCompleteDayPairs(self, hours: List[int]) -> int:
        n = len(hours)
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                if (hours[i] + hours[j]) % 24 == 0:
                    count += 1
        return count

# Example usage:
if __name__ == "__main__":
    sol = Solution()
    print(sol.countCompleteDayPairs([12,12,30,24,24]))  # Output: 2
    print(sol.countCompleteDayPairs([72,48,24,3]))      # Output: 3