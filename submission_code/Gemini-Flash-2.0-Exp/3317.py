class Solution:
    def maxPalindromesAfterOperations(self, words: List[str]) -> int:
        counts = {}
        for word in words:
            for char in word:
                counts[char] = counts.get(char, 0) + 1
        
        odd_count = 0
        for char in counts:
            if counts[char] % 2 != 0:
                odd_count += 1
        
        lengths = sorted([len(word) for word in words])
        
        ans = 0
        for length in lengths:
            if length % 2 == 0:
                if odd_count >= 2:
                    odd_count -= 2
                    ans += 0
                elif odd_count == 1:
                    if length <= 1:
                        continue
                    else:
                        odd_count = 1
                        break
                else:
                    ans += 1
            else:
                if odd_count >= 1:
                    odd_count -= 1
                    ans += 1
                else:
                    break
        
        return ans