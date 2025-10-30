from collections import Counter

class Solution:
    def minimumDeletions(self, word: str, k: int) -> int:
        freq = list(Counter(word).values())
        original_length = sum(freq)
        if not freq:
            return 0  # According to constraints, word is non-empty
        
        critical_points = set()
        max_m_candidate = 0
        for f in freq:
            critical_points.add(f)
            current = f + k
            critical_points.add(current)
            if current > max_m_candidate:
                max_m_candidate = current
        
        critical_points.add(0)
        critical_points.add(max_m_candidate)
        
        max_sum = 0
        for m in critical_points:
            current_sum = 0
            for f in freq:
                if f >= m - k:
                    current_sum += min(f, m)
            if current_sum > max_sum:
                max_sum = current_sum
        
        return original_length - max_sum