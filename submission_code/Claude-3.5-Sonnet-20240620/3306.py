class Solution:
    def unmarkedSumArray(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        n = len(nums)
        marked = [False] * n
        unmarked_sum = sum(nums)
        unmarked = [(num, i) for i, num in enumerate(nums)]
        unmarked.sort()
        
        answer = []
        unmarked_index = 0
        
        for index, k in queries:
            if not marked[index]:
                marked[index] = True
                unmarked_sum -= nums[index]
            
            while k > 0 and unmarked_index < n:
                num, i = unmarked[unmarked_index]
                if not marked[i]:
                    marked[i] = True
                    unmarked_sum -= num
                    k -= 1
                unmarked_index += 1
            
            answer.append(unmarked_sum)
        
        return answer