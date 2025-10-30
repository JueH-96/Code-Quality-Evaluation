class Solution:
    def maxOperations(self, nums: List[int]) -> int:
        n = len(nums)
        if n < 2:
            return 0

        max_ops = 0
        for i in range(1 << n):
            current_nums = nums[:]
            ops = 0
            scores = []
            
            temp_ops = 0
            while len(current_nums) >= 2:
                score = current_nums[0] + current_nums[1]
                current_nums = current_nums[2:]
                temp_ops +=1
                scores.append(score)

            if len(scores) > 0:
                first_score = scores[0]
                all_same = all(s == first_score for s in scores)
                if all_same:
                    max_ops = max(max_ops, len(scores))

        return max_ops