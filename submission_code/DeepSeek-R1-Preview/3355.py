class Solution:
    def minimumLevels(self, possible: List[int]) -> int:
        n = len(possible)
        prefix = [0]
        for num in possible:
            prefix.append(prefix[-1] + (1 if num else -1))
        total = prefix[-1]
        min_k = float('inf')
        for k in range(1, n):
            alice = prefix[k]
            bob = total - alice
            if alice > bob:
                if k < min_k:
                    min_k = k
        return min_k if min_k != float('inf') else -1