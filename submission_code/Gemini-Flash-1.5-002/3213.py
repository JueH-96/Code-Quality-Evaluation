class Solution:
    def countSubarrays(self, nums: List[int], k: int) -> int:
        count = 0
        max_freq = 0
        max_num = -1
        freq_map = {}
        for i in range(len(nums)):
            freq_map = {}
            for j in range(i, len(nums)):
                if nums[j] not in freq_map:
                    freq_map[nums[j]] = 0
                freq_map[nums[j]] += 1
                max_freq_curr = 0
                max_num_curr = -1
                for num, freq in freq_map.items():
                    if freq > max_freq_curr:
                        max_freq_curr = freq
                        max_num_curr = num
                    elif freq == max_freq_curr and num < max_num_curr:
                        max_num_curr = num

                if max_freq_curr >=k:
                    count +=1

        return count