class Solution:
    def minimumSum(self, n: int, k: int) -> int:
        chosen = []
        current = 1
        while len(chosen) < n:
            # Check if current can be added without forming a pair summing to k
            if all(current + x != k for x in chosen):
                chosen.append(current)
            current += 1
        return sum(chosen)