class Solution:
    def findWinningPlayer(self, skills: List[int], k: int) -> int:
        n = len(skills)
        prefix_skills = [0] * (n + 1)
        suffix_skills = [0] * (n + 1)
        for i in range(n):
            prefix_skills[i + 1] = prefix_skills[i] + skills[i]
            suffix_skills[n - i] = suffix_skills[n - i + 1] + skills[n - i - 1]
        for i in range(n):
            if prefix_skills[i] > suffix_skills[i + 1]:
                if (i + 1) % k == 0:
                    return 0
            elif prefix_skills[i] < suffix_skills[i + 1]:
                if (n - i) % k == 0:
                    return 1
        return -1