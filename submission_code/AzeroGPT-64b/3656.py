class Solution:
    def minimumOperations(self, nums: List[int]) -> int:
        if len(nums) <= 1:
            return 0
        min_ops = 0
        count_map = {}
        for val in nums:
            if val in count_map:
                count_map[val] += 1
            else:
                count_map[val] = 1
        val = [(count_map.get(i, 0), i) for i in range(1,101)]
        top3 = sorted(val, reverse=True)[:3]
        remain = 0
        while len(set(top3)) != len(top3):
            remain = top3[2][0]
            top3 = [t if t[0] != remain else (0,0) for t in top3]
            top3 = sorted(top3, reverse=True)[:3]
        return -(-len(nums) // 3) if min([t[0] for t in top3]) == remain else  len(nums) // 3