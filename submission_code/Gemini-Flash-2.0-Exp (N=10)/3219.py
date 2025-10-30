class Solution:
    def lexicographicallySmallestArray(self, nums: List[int], limit: int) -> List[int]:
        n = len(nums)
        visited = [False] * n
        
        for i in range(n):
            if visited[i]:
                continue
            
            group = []
            q = [i]
            visited[i] = True
            
            while q:
                curr = q.pop(0)
                group.append(curr)
                
                for j in range(i + 1, n):
                    if not visited[j] and abs(nums[curr] - nums[j]) <= limit:
                        visited[j] = True
                        q.append(j)
            
            values = [nums[idx] for idx in group]
            values.sort()
            group.sort()
            
            for k in range(len(group)):
                nums[group[k]] = values[k]
        
        return nums