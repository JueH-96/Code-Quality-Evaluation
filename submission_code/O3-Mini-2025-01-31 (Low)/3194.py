from typing import List

class Solution:
    def findWordsContaining(self, words: List[str], x: str) -> List[int]:
        # Create an empty list to store indices of words containing character x
        indices = []
        # Iterate through the words using enumerate to get both index and word.
        for i, word in enumerate(words):
            if x in word:
                indices.append(i)
        return indices

# Example usage:
if __name__ == "__main__":
    solution = Solution()
    # Example 1
    print(solution.findWordsContaining(["leet", "code"], "e"))  # Output: [0, 1]
    # Example 2
    print(solution.findWordsContaining(["abc", "bcd", "aaaa", "cbc"], "a"))  # Output: [0, 2]
    # Example 3
    print(solution.findWordsContaining(["abc", "bcd", "aaaa", "cbc"], "z"))  # Output: []