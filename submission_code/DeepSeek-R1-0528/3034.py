class Solution:
    def numberOfPoints(self, nums: List[List[int]]) -> int:
        nums.sort(key=lambda x: x[0])
        merged = []
        for start, end in nums:
            if not merged:
                merged.append([start, end])
            else:
                last_end = merged[-1][1]
                if start <= last_end + 1:
                    merged[-1][1] = max(last_end, end)
                else:
                    merged.append([start, end])
        return sum(end - start + 1 for start, end in merged)