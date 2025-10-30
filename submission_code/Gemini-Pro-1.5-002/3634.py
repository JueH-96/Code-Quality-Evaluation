class Solution:
    def calculateScore(self, s: str) -> int:
        n = len(s)
        marked = [False] * n
        score = 0

        for i in range(n):
            if marked[i]:
                continue

            best_j = -1
            min_dist = float('inf')

            for j in range(i):
                if not marked[j] and ord(s[i]) + ord(s[j]) == ord('a') + ord('z'):
                    if i - j < min_dist:
                        min_dist = i - j
                        best_j = j
            
            if best_j != -1:
                marked[i] = True
                marked[best_j] = True
                score += i - best_j

        return score