class Solution:
    def minTime(self, skill: List[int], mana: List[int]) -> int:
        n = len(skill)
        m = len(mana)
        if m == 0:
            return 0
        prefix_arr = [0] * (n + 1)
        for i in range(n):
            prefix_arr[i + 1] = prefix_arr[i] + skill[i]
        F_arr = [0] * n
        first_mana = mana[0]
        for i in range(n):
            F_arr[i] = prefix_arr[i + 1] * first_mana
        
        for j in range(1, m):
            manaj = mana[j]
            Tj_val = -10**18
            for i in range(n):
                val = F_arr[i] - prefix_arr[i] * manaj
                if val > Tj_val:
                    Tj_val = val
            for i in range(n):
                F_arr[i] = Tj_val + prefix_arr[i + 1] * manaj
        
        return F_arr[n - 1]