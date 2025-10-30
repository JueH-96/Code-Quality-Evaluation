from collections import Counter

class Solution:
    def makeStringGood(self, s: str) -> int:
        counts = Counter(s)
        letters = 'abcdefghijklmnopqrstuvwxyz'
        cnts = [counts.get(c, 0) for c in letters]
        
        min_ops = float('inf')
        total = sum(cnts)
        max_k = max(cnts) if cnts else 0
        
        # Consider k up to the maximum possible, which could be up to total
        # but to limit computation, we can check up to max_k + 1
        for k in range(0, max_k + 2):
            cost = 0
            carry = 0
            for c in cnts:
                current = c + carry
                if current >= k:
                    surplus = current - k
                    cost += surplus
                    carry = surplus
                else:
                    deficit = k - current
                    cost += deficit
                    carry = 0
            # Any remaining carry after 'z' is deleted
            cost += carry
            min_ops = min(min_ops, cost)
        
        # Check for cases where k is larger than the maximum count, requiring insertions
        # We check a few values beyond max_k to ensure optimality
        for additional_k in range(1, 3):
            k = max_k + additional_k
            cost = 0
            carry = 0
            for c in cnts:
                current = c + carry
                if current >= k:
                    surplus = current - k
                    cost += surplus
                    carry = surplus
                else:
                    deficit = k - current
                    cost += deficit
                    carry = 0
            cost += carry
            min_ops = min(min_ops, cost)
        
        return min_ops