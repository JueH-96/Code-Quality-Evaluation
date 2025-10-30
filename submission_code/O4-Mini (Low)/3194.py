from typing import List

class Solution:
    def findWordsContaining(self, words: List[str], x: str) -> List[int]:
        """
        Returns the list of indices of words that contain the character x.
        """
        result = []
        for i, word in enumerate(words):
            if x in word:
                result.append(i)
        return result

# Example usage:
if __name__ == "__main__":
    sol = Solution()
    print(sol.findWordsContaining(["leet","code"], "e"))  # Output: [0, 1]
    print(sol.findWordsContaining(["abc","bcd","aaaa","cbc"], "a"))  # Output: [0, 2]
    print(sol.findWordsContaining(["abc","bcd","aaaa","cbc"], "z"))  # Output: []