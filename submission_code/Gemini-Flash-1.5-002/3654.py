class Solution:
    def minArraySum(self, nums: List[int], k: int, op1: int, op2: int) -> int:
        n = len(nums)
        ops = []
        for i, num in enumerate(nums):
            ops_i = []
            #op1
            temp = num
            count1 = 0
            while count1 < op1 and temp > 0:
                temp = (temp + 1) // 2
                count1 +=1
                ops_i.append((temp,count1,0)) # (value, op1_count, op2_count)

            #op2
            temp = num
            count2 = 0
            while count2 < op2 and temp >=k:
                temp -=k
                count2 +=1
                ops_i.append((temp,0,count2))

            #op1 then op2
            temp = num
            count1 = 0
            count2 = 0
            while count1 < op1 and temp > 0:
                temp = (temp + 1) // 2
                count1 +=1
                temp2 = temp
                while count2 < op2 and temp2 >=k:
                    temp2 -=k
                    count2 +=1
                    ops_i.append((temp2,count1,count2))
            
            #op2 then op1
            temp = num
            count1 = 0
            count2 = 0
            while count2 < op2 and temp >=k:
                temp -=k
                count2 +=1
                temp2 = temp
                while count1 < op1 and temp2 > 0:
                    temp2 = (temp2 + 1) // 2
                    count1 +=1
                    ops_i.append((temp2,count1,count2))
            ops_i.append((num,0,0))
            ops.append(ops_i)

        
        dp = {}
        def solve(idx, op1_left, op2_left):
            if idx == n:
                return 0
            if (idx, op1_left, op2_left) in dp:
                return dp[(idx, op1_left, op2_left)]
            
            min_sum = float('inf')
            for op in ops[idx]:
                val, op1_count, op2_count = op
                if op1_count <= op1_left and op2_count <= op2_left:
                    min_sum = min(min_sum, val + solve(idx + 1, op1_left - op1_count, op2_left - op2_count))
            dp[(idx, op1_left, op2_left)] = min_sum
            return min_sum

        return solve(0, op1, op2)