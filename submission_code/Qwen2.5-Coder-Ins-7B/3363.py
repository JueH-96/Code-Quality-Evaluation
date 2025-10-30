class Solution:
    def mostFrequentIDs(self, nums: List[int], freq: List[int]) -> List[int]:
        from sortedcontainers import SortedList
        
        id_freq = {}
        freq_list = SortedList()
        result = []
        
        for i in range(len(nums)):
            if nums[i] in id_freq:
                freq_list.remove(id_freq[nums[i]])
            id_freq[nums[i]] += freq[i]
            freq_list.add(id_freq[nums[i]])
            result.append(freq_list[-1] if freq_list else 0)
        
        return result