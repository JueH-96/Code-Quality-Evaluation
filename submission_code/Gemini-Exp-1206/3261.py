class Solution:
    def minOrAfterOperations(self, nums: List[int], k: int) -> int:
        ans = 0
        mask = 0
        for bit in range(29, -1, -1):
            mask |= 1 << bit
            pref = 0
            cnt = 0
            temp = []
            for x in nums:
                pref &= x
                if (pref & mask) == 0:
                    temp.append(pref)
                    pref = (1 << 30) - 1
                else:
                    cnt += 1
            if pref != (1 << 30) - 1:
                temp.append(pref)

            if len(nums) - len(temp) > k:
                ans |= 1 << bit
                mask ^= 1 << bit

        return ans