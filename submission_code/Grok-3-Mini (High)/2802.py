class Solution:
    def punishmentNumber(self, n: int) -> int:
        def can_sum(s: str, target: int, index: int, memo: dict) -> bool:
            if index == len(s):
                return target == 0
            if target < 0:
                return False
            key = (index, target)
            if key in memo:
                return memo[key]
            for len_sub in range(1, len(s) - index + 1):
                sub_str = s[index : index + len_sub]
                try:
                    val = int(sub_str)
                    if can_sum(s, target - val, index + len_sub, memo):
                        memo[key] = True
                        return True
                except ValueError:
                    continue
            memo[key] = False
            return False
        
        ans = 0
        for i in range(1, n + 1):
            sq = i * i
            s_sq = str(sq)
            memo = {}
            if can_sum(s_sq, i, 0, memo):
                ans += sq
        return ans