class Solution:
    def minTime(self, skill: List[int], mana: List[int]) -> int:
        n = len(skill)
        m = len(mana)
        
        # Compute prefix sums
        prefix = [0] * n
        prefix[0] = skill[0]
        for i in range(1, n):
            prefix[i] = prefix[i-1] + skill[i]
        
        T = [0] * m
        T[0] = 0
        
        for j in range(1, m):
            # Constraint from wizard 0
            T[j] = T[j-1] + skill[0] * mana[j-1]
            
            # Constraints from other wizards
            for i in range(1, n):
                constraint = T[j-1] + prefix[i-1] * (mana[j-1] - mana[j]) + skill[i] * mana[j-1]
                T[j] = max(T[j], constraint)
        
        # The answer is when the last wizard finishes the last potion
        return T[m-1] + prefix[n-1] * mana[m-1]