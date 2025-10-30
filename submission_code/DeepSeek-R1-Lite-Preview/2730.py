class Solution:
    def maximumOr(self, nums: List[int], k: int) -> int:
        min_m = [k + 1] * 31  # Initialize with k+1, which is greater than k

        # Find set bits for each number and update min_m
        for num in nums:
            t = 0
            temp = num
            while temp > 0:
                if temp & 1:
                    # Set bit t is set in num
                    for b in range(t, min(t + k + 1, 31)):
                        min_m[b] = min(min_m[b], b - t)
                temp >>= 1
                t += 1

        # First, set all bits where min_m[b] == 0
        or_result = 0
        for b in range(31):
            if min_m[b] == 0:
                or_result |= (1 << b)

        # Now, sort remaining bits in descending order and try to set them with remaining k
        remaining_bits = sorted([b for b in range(31) if min_m[b] > 0], reverse=True)
        rem_k = k
        for b in remaining_bits:
            if min_m[b] <= rem_k:
                or_result |= (1 << b)
                rem_k -= min_m[b]

        return or_result