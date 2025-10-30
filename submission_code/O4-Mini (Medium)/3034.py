class Solution:
    def numberOfPoints(self, nums: List[List[int]]) -> int:
        # Since all start_i and end_i are between 1 and 100,
        # we can mark each covered integer point in a boolean array.
        covered = [False] * 101  # indices 0..100
        
        for start, end in nums:
            for point in range(start, end + 1):
                covered[point] = True
        
        # Count how many points are covered
        return sum(covered)