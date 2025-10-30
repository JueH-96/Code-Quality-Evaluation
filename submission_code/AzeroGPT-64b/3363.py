class Solution:
    def mostFrequentIDs(self, nums: List[int], freq: List[int]) -> List[int]:
        id_count = Counter()
        max_count = 0
        result = []

        for i, (id_value, change) in enumerate(zip(nums, freq)):
            id_count[id_value] += change
            if id_count[id_value] > max_count:
                max_count = id_count[id_value]
            elif not id_count[id_value]:
                del id_count[id_value]

            if max_count and any(id_count[id_] == max_count for id_ in id_count):
                result.append(max_count)
            else:
                result.append(0)
        
        return result