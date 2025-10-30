from typing import List

class Solution:
    def countPrefixSuffixPairs(self, words: List[str]) -> int:
        def isPrefixAndSuffix(prefix: str, s: str) -> bool:
            # Check if prefix is actually valid (i.e., it must be shorter than or equal to s)
            if len(prefix) > len(s):
                return False
            return s.startswith(prefix) and s.endswith(prefix)
        
        count = 0
        n = len(words)
        for i in range(n):
            for j in range(i+1, n):
                if isPrefixAndSuffix(words[i], words[j]):
                    count += 1
        return count

# Example usage:
if __name__ == "__main__":
    sol = Solution()
    # Test case 1:
    words1 = ["a", "aba", "ababa", "aa"]
    print(sol.countPrefixSuffixPairs(words1))  # Output: 4

    # Test case 2:
    words2 = ["pa", "papa", "ma", "mama"]
    print(sol.countPrefixSuffixPairs(words2))  # Output: 2

    # Test case 3:
    words3 = ["abab", "ab"]
    print(sol.countPrefixSuffixPairs(words3))  # Output: 0