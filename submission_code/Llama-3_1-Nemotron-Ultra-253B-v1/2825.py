class Solution:
    def minimizedStringLength(self, s: str) -> int:
        s = list(s)
        while True:
            max_deletions = 0
            best_i = -1
            best_left = -1
            best_right = -1
            for i in range(len(s)):
                c = s[i]
                left = -1
                for j in range(i-1, -1, -1):
                    if s[j] == c:
                        left = j
                        break
                right = -1
                for j in range(i+1, len(s)):
                    if s[j] == c:
                        right = j
                        break
                deletions = 0
                if left != -1:
                    deletions += 1
                if right != -1:
                    deletions += 1
                if deletions > max_deletions or (deletions == max_deletions and best_i == -1):
                    max_deletions = deletions
                    best_i = i
                    best_left = left
                    best_right = right
            if max_deletions == 0:
                break
            new_s = []
            for j in range(len(s)):
                if j == best_left or j == best_right:
                    continue
                new_s.append(s[j])
            s = new_s
        return len(s)