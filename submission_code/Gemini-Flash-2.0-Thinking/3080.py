class Solution:
    def maxSubarrays(self, nums: List[int]) -> int:
        n = len(nums)
        if not nums:
            return 0

        min_sum_score = -1

        def calculate_score(subarray):
            if not subarray:
                return -1
            score = subarray[0]
            for i in range(1, len(subarray)):
                score &= subarray[i]
            return score

        def get_sum_of_scores(split):
            total_score = 0
            for subarray_indices in split:
                subarray = [nums[i] for i in subarray_indices]
                total_score += calculate_score(subarray)
            return total_score

        def find_min_sum_score():
            min_score = float('inf')
            for i in range(1, 1 << (n - 1)):
                split = []
                current_subarray = []
                for j in range(n):
                    current_subarray.append(j)
                    if (i >> j) & 1 or j == n - 1:
                        split.append(tuple(current_subarray))
                        current_subarray = []

                if current_subarray:
                    split.append(tuple(current_subarray))

                total_score = get_sum_of_scores(split)
                min_score = min(min_score, total_score)
            
            # Handle the case of a single subarray
            min_score = min(min_score, calculate_score(nums))

            return min_score

        min_sum_score_optimized = -1
        current_and = nums[0]
        for i in range(1, n):
            current_and &= nums[i]
        min_sum_score_optimized = current_and

        max_subarrays_count = 0
        min_sum = float('inf')

        for i in range(1, 1 << (n - 1)):
            split_indices = []
            start = 0
            for j in range(n - 1):
                if (i >> j) & 1:
                    split_indices.append((start, j + 1))
                    start = j + 1 + 1
            split_indices.append((start, n))

            split = []
            current_start = 0
            for end in range(1, n + 1):
                # Check if this is a split point
                if end == n or (i >> (end - 1)) & 1:
                    split.append(nums[current_start:end])
                    current_start = end

            current_sum_score = 0
            num_subarrays = len(split)
            for sub_array in split:
                if sub_array:
                    score = sub_array[0]
                    for k in range(1, len(sub_array)):
                        score &= sub_array[k]
                    current_sum_score += score

            if current_sum_score < min_sum:
                min_sum = current_sum_score
                max_subarrays_count = num_subarrays
            elif current_sum_score == min_sum:
                max_subarrays_count = max(max_subarrays_count, num_subarrays)

        # Optimized approach based on minimizing sum to 0
        count = 0
        current_and = -1
        for num in nums:
            if current_and == -1:
                current_and = num
            else:
                current_and &= num
            if current_and == 0:
                count += 1
                current_and = -1

        if count == 0:
            return 1
        else:
            return count