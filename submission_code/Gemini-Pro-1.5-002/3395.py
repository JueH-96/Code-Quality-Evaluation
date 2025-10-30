class Solution:
    def minAnagramLength(self, s: str) -> int:
        n = len(s)
        if n % 2 != 0:
            return n
        
        s1 = s[:n//2]
        s2 = s[n//2:]
        
        count1 = {}
        count2 = {}
        
        for char in s1:
            count1[char] = count1.get(char, 0) + 1
        
        for char in s2:
            count2[char] = count2.get(char, 0) + 1
            
        diff = 0
        for char in count1:
            if char in count2:
                diff += abs(count1[char] - count2[char])
                
        
        
        ans = 0
        for char in count1:
            if char not in count2:
                ans += count1[char]
        for char in count2:
            if char not in count1:
                ans += count2[char]
        
        return abs(n//2 - (sum(count1.values()) - (diff //2) ))