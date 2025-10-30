class Solution:
    def canSplitArray(self, nums: List[int], m: int) -> bool:
        n = len(nums)
        if n <= 1:
            return True
        
        def can_split(arr):
            if len(arr) == 1:
                return True
            
            for i in range(1, len(arr)):
                left_arr = arr[:i]
                right_arr = arr[i:]
                
                left_valid = len(left_arr) == 1 or sum(left_arr) >= m
                right_valid = len(right_arr) == 1 or sum(right_arr) >= m
                
                if left_valid and right_valid:
                    return True
            return False

        
        q = [nums]
        visited = {tuple(nums)}
        
        while q:
            curr_arr = q.pop(0)
            if len(curr_arr) == 1:
                continue
            
            
            split_found = False
            for i in range(1, len(curr_arr)):
                left_arr = curr_arr[:i]
                right_arr = curr_arr[i:]
                
                left_valid = len(left_arr) == 1 or sum(left_arr) >= m
                right_valid = len(right_arr) == 1 or sum(right_arr) >= m
                
                if left_valid and right_valid:
                    split_found = True
                    if len(left_arr) > 1 and tuple(left_arr) not in visited:
                        q.append(left_arr)
                        visited.add(tuple(left_arr))
                    if len(right_arr) > 1 and tuple(right_arr) not in visited:
                        q.append(right_arr)
                        visited.add(tuple(right_arr))
            if not split_found:
                return False
        
        return True