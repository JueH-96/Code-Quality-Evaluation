class Solution:
    def numberGame(self, nums: List[int]) -> List[int]:
        arr = []
        while nums:
            # Alice's turn: remove the minimum
            a = min(nums)
            nums.remove(a)
            if not nums:
                break
            # Bob's turn: remove the new minimum
            b = min(nums)
            nums.remove(b)
            # Append Bob's element first, then Alice's
            arr.append(b)
            arr.append(a)
        return arr