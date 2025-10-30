class Solution:
    def maxPartitionsAfterOperations(self, s: str, k: int) -> int:
        def compute_partitions(s, k):
            partitions = []
            n = len(s)
            left = 0
            while left < n:
                freq = {}
                distinct = 0
                right = left
                while right < n and distinct <= k:
                    char = s[right]
                    if char not in freq or freq[char] == 0:
                        distinct += 1
                        freq[char] = 0
                    freq[char] += 1
                    if distinct > k:
                        break
                    right += 1
                partitions.append(s[left:right])
                left = right
            return len(partitions)
        
        original = compute_partitions(s, k)
        max_result = original
        
        for i in range(len(s)):
            for c in 'abcdefghijklmnopqrstuvwxyz':
                if c == s[i]:
                    continue
                modified = s[:i] + c + s[i+1:]
                current = compute_partitions(modified, k)
                if current > max_result:
                    max_result = current
        
        return max_result