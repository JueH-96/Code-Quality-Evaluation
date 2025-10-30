class Solution:
    def maxPalindromesAfterOperations(self, words: List[str]) -> int:
        count = 0
        char_counts = {}
        for word in words:
            odd_chars = 0
            char_count = {}
            for char in word:
                char_count[char] = char_count.get(char, 0) + 1
            for c in char_count:
                if char_count[c] % 2 != 0:
                    odd_chars += 1
            if odd_chars == 0 or odd_chars % 2 == 0:
                count += 1
            else:
                for char in char_count:
                    if char_count[char] %2 !=0:
                        if char in char_counts and char_counts[char] > 0:
                            char_counts[char] -=1
                            count +=1
                            break
                        else:
                            char_counts[char] = char_counts.get(char,0) +1
                            break

        return count