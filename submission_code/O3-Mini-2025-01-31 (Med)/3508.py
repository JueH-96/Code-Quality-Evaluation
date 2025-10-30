class Solution:
    def minChanges(self, n: int, k: int) -> int:
        # We can only change '1's to '0's.
        # So, for every bit position where k has 1, n must also have 1.
        # If not, it's impossible to achieve k.
        if (n & k) != k:
            return -1
        
        # The minimum changes needed are the extra 1's in n that are not needed in k.
        # That is, count the number of 1's in n minus the number of 1's in k.
        return bin(n).count("1") - bin(k).count("1")

# Simple tests
if __name__ == "__main__":
    solution = Solution()
    print(solution.minChanges(13, 4))   # Output: 2
    print(solution.minChanges(21, 21))  # Output: 0
    print(solution.minChanges(14, 13))  # Output: -1