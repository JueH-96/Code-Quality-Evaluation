from typing import List

class Solution:
    def minTime(self, skill: List[int], mana: List[int]) -> int:
        n = len(skill)
        m = len(mana)
        # prefix sums of skill: A[i] = sum(skill[0..i])
        A = [0] * n
        s = 0
        for i in range(n):
            s += skill[i]
            A[i] = s

        # S will accumulate the start time of each potion on wizard 0
        S = 0
        # For each new potion j (from 1 to m-1), compute the required delay D_j
        # so that no blocking occurs on any wizard.
        for j in range(1, m):
            prev_b = mana[j - 1]
            curr_b = mana[j]
            # i = 0 case
            best = A[0] * prev_b
            # i >= 1 cases
            # t_i = prev_b * A[i] - curr_b * A[i-1]
            # take max over i
            # use local vars for speed
            Ai_prev = A[0]
            for i in range(1, n):
                Ai = A[i]
                # compute t_i
                t = prev_b * Ai - curr_b * Ai_prev
                if t > best:
                    best = t
                Ai_prev = Ai
            # delay start of potion j by best
            S += best

        # finally add processing of last potion on all wizards
        # that's sum(skill) * mana[m-1] = A[n-1] * mana[m-1]
        return S + A[-1] * mana[-1]