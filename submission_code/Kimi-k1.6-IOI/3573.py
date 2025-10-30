from collections import Counter

class Solution:
    def validSubstringCount(self, word1: str, word2: str) -> int:
        needed_counts = Counter(word2)
        L = len(word2)
        n = len(word1)
        total_counts = Counter(word1)
        
        # Check if word1 has at least the required counts for all characters in word2
        for c in needed_counts:
            if total_counts.get(c, 0) < needed_counts[c]:
                return 0
        
        # Precompute prefix sums for each character
        prefix = [[0] * 26]
        for c in word1:
            prev = prefix[-1].copy()
            prev[ord(c) - ord('a')] += 1
            prefix.append(prev)
        
        required_chars = list(needed_counts.keys())
        answer = 0
        
        # Iterate over each possible starting index i
        for i in range(n - L + 1):
            low = i + L - 1
            high = n - 1
            ans_j = -1
            left, right = low, high
            
            while left <= right:
                mid = (left + right) // 2
                valid = True
                for c in required_chars:
                    idx = ord(c) - ord('a')
                    current_count = prefix[mid + 1][idx] - prefix[i][idx]
                    if current_count < needed_counts[c]:
                        valid = False
                        break
                if valid:
                    ans_j = mid
                    right = mid - 1
                else:
                    left = mid + 1
            
            if ans_j != -1:
                answer += (n - ans_j)
        
        return answer