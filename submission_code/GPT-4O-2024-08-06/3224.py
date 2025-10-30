class Solution:
    def numberOfSequence(self, n: int, sick: List[int]) -> int:
        MOD = 10**9 + 7

        # Calculate the number of uninfected children between each pair of sick children
        gaps = []
        prev_sick = -1
        for s in sick:
            if prev_sick != -1:
                gaps.append(s - prev_sick - 1)
            prev_sick = s

        # Add the gaps before the first sick child and after the last sick child
        if sick[0] > 0:
            gaps.insert(0, sick[0])
        if sick[-1] < n - 1:
            gaps.append(n - 1 - sick[-1])

        # Calculate the number of ways to infect each gap
        def factorial(x):
            result = 1
            for i in range(2, x + 1):
                result = (result * i) % MOD
            return result

        result = 1
        for gap in gaps:
            if gap > 0:
                result = (result * factorial(gap)) % MOD

        return result