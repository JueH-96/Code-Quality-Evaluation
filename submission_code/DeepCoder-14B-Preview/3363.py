class Solution:
    def mostFrequentIDs(self, nums: List[int], freq: List[int]) -> List[int]:
        count = dict()  # Maps ID to its current count
        freq_map = dict()  # Maps count to the number of IDs with that count
        current_max = 0
        ans = []
        
        for i in range(len(nums)):
            id = nums[i]
            delta = freq[i]
            
            # Current count of the ID
            current_count = count.get(id, 0)
            
            # New count after applying delta
            new_count = current_count + delta
            
            # Update freq_map for the current_count if it's positive
            if current_count > 0:
                if current_count in freq_map:
                    freq_map[current_count] -= 1
                    if freq_map[current_count] == 0:
                        del freq_map[current_count]
            
            # Update count and freq_map for the new_count
            if new_count > 0:
                count[id] = new_count
                if new_count in freq_map:
                    freq_map[new_count] += 1
                else:
                    freq_map[new_count] = 1
            else:
                if id in count:
                    del count[id]
            
            # Update current_max
            if new_count > current_max:
                current_max = new_count
            else:
                if current_max not in freq_map:
                    if freq_map:
                        current_max = max(freq_map.keys())
                    else:
                        current_max = 0
            
            # Append the result for this step
            if freq_map:
                ans.append(current_max)
            else:
                ans.append(0)
        
        return ans