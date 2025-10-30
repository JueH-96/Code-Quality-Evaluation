class Solution:
    def minAnagramLength(self, s: str) -> int:
        n = len(s)
        
        def char_count(substring):
            count = {}
            for char in substring:
                count[char] = count.get(char, 0) + 1
            return count
        
        for length in range(1, n + 1):
            if n % length == 0:
                chunks = [s[i:i+length] for i in range(0, n, length)]
                first_count = char_count(chunks[0])
                if all(char_count(chunk) == first_count for chunk in chunks):
                    return length
        
        return n