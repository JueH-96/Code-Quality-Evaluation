class Solution:
    def earliestSecondToMarkIndices(self, nums: List[int], changeIndices: List[int]) -> int:
        n = len(nums)
        m = len(changeIndices)
        present_indices = set(changeIndices)
        if len(present_indices) < n:
            return -1
            
        def is_possible(seconds):
            last_occurrence = {}
            for s in range(seconds):
                last_occurrence[changeIndices[s]] = s + 1
            
            if len(last_occurrence) < n:
                return False

            current_nums = list(nums)
            marked = [False] * n
            
            seconds_ops = [False] * seconds
            for s in range(seconds):
                index_to_check = changeIndices[s]
                if last_occurrence[index_to_check] == s + 1:
                    seconds_ops[s] = True # Mark operation at this second if possible

            for s in range(seconds):
                index_to_check = changeIndices[s]
                if seconds_ops[s]:
                    if current_nums[index_to_check-1] == 0:
                        marked[index_to_check-1] = True
                    else:
                        # Need to decrement before this second to make it 0
                        needed_decrement = current_nums[index_to_check-1]
                        decrement_seconds_available = 0
                        decrement_indices = []
                        for prev_s in range(s):
                            if not seconds_ops[prev_s]:
                                decrement_seconds_available += 1
                                decrement_indices.append(prev_s)
                        
                        if decrement_seconds_available < needed_decrement:
                            return False
                        
                        for i in range(needed_decrement):
                            decrement_s_index = decrement_indices[i]
                            current_nums[index_to_check-1] -= 1
                        
                        if current_nums[index_to_check-1] == 0:
                            marked[index_to_check-1] = True
                        else:
                            return False
                else:
                    # Decrement operation
                    earliest_unmarked_index = -1
                    earliest_last_time = float('inf')
                    for i in range(n):
                        if not marked[i]:
                            if last_occurrence.get(i+1, float('inf')) < earliest_last_time:
                                earliest_last_time = last_occurrence.get(i+1, float('inf'))
                                earliest_unmarked_index = i + 1
                    if earliest_unmarked_index != -1:
                        current_nums[earliest_unmarked_index-1] = max(0, current_nums[earliest_unmarked_index-1] - 1)

            return all(marked)

        low = 1
        high = m
        ans = -1
        while low <= high:
            mid = (low + high) // 2
            if is_possible(mid):
                ans = mid
                high = mid - 1
            else:
                low = mid + 1
        return ans