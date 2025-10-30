class Solution:
    def minimumMoves(self, nums: List[int], k: int, maxChanges: int) -> int:
        # First, find all the positions of 1s in the array
        ones = [i for i, num in enumerate(nums) if num == 1]
        n = len(ones)
        if n >= k:
            # If there are enough 1s, we can pick k of them directly
            # The minimal moves is the sum of distances from the median
            # to the k nearest 1s
            # To find the minimal sum of distances, we can use the median
            # as the pivot
            # We need to find the k consecutive 1s with the smallest sum of distances
            # to their median
            # So we can use a sliding window of size k
            # Calculate the sum of distances for each window and find the minimal one
            min_moves = float('inf')
            # Precompute the prefix sum of the positions
            prefix = [0] * (n + 1)
            for i in range(n):
                prefix[i+1] = prefix[i] + ones[i]
            # Iterate over all possible windows of size k
            for i in range(n - k + 1):
                # The median is the middle element in the window
                median = ones[i + k // 2]
                # Calculate the sum of absolute differences from the median
                # sum_{j=i}^{i+k-1} |ones[j] - median|
                # Using the prefix sum, we can compute this efficiently
                # sum_{j=i}^{i+k-1} ones[j] = prefix[i+k] - prefix[i]
                # sum_{j=i}^{i+k-1} |ones[j] - median| = sum_{j=i}^{i+k-1} (ones[j] - median) if ones[j] >= median
                # else sum_{j=i}^{i+k-1} (median - ones[j])
                # To compute this, we can find the number of elements less than median and greater than median
                # and use the prefix sum to compute the sum
                # Find the index where median is in the window
                # Since the window is sorted, the median is at i + k//2
                # So the number of elements less than median is k//2
                # and the number of elements greater than median is k - k//2 - 1
                # sum_{j=i}^{i+k-1} |ones[j] - median| = (median * (k//2) - sum_{j=i}^{i+k//2-1} ones[j]) + (sum_{j=i+k//2+1}^{i+k-1} ones[j] - median * (k - k//2 - 1))
                left_sum = prefix[i + k//2] - prefix[i]
                right_sum = prefix[i + k] - prefix[i + k//2 + 1]
                moves = (median * (k//2) - left_sum) + (right_sum - median * (k - k//2 - 1))
                if moves < min_moves:
                    min_moves = moves
            return min_moves
        else:
            # If there are not enough 1s, we need to use maxChanges to create new 1s
            # The number of 1s we need to create is k - n
            # Each creation requires one move
            # Then, we need to move these new 1s to the aliceIndex
            # The minimal moves is the number of creations plus the sum of distances
            # from the aliceIndex to the positions of the new 1s
            # To minimize the sum of distances, we should place the new 1s as close as possible to the aliceIndex
            # So, the minimal moves is (k - n) * 1 + (k - n) * 0 = k - n
            # But we need to consider that we can also use the existing 1s
            # So, the total moves is (k - n) + sum of distances from aliceIndex to the existing 1s
            # To minimize the sum of distances, we should choose the aliceIndex as the median of the existing 1s
            # So, first, find the median of the existing 1s
            if n == 0:
                # If there are no 1s, we need to create k 1s and move them to the aliceIndex
                # The minimal moves is k * 1 + k * 0 = k
                return k
            median = ones[n // 2]
            # Calculate the sum of distances from the median to all existing 1s
            sum_dist = 0
            for pos in ones:
                sum_dist += abs(pos - median)
            # The total moves is (k - n) + sum_dist
            return (k - n) + sum_dist