class Solution:
    def isPossibleToSplit(self, nums: List[int]) -> bool:
        n = len(nums)
        if n == 0 or n % 2 != 0:
            return False

        count = {}
        for num in nums:
            count[num] = count.get(num, 0) + 1

        duplicates = []
        uniques = []
        for num, freq in count.items():
            if freq > 1:
                duplicates.append(num)
            else:
                uniques.append(num)

        num_duplicates = len(duplicates)
        num_uniques = len(uniques)
        half_len = n // 2

        if num_uniques >= half_len:
            return True

        
        remaining_spots = half_len - num_uniques
        if num_duplicates >= remaining_spots:
            return True
        else:
            return False