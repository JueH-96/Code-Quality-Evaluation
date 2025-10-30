from collections import defaultdict

class Solution:
    def lexicographicallySmallestArray(self, nums: List[int], limit: int) -> List[int]:
        def merge_sort(arr, index_map, start, end):
            if end - start <= 1:
                return arr[start:end], index_map[start:end]

            mid = (start + end) // 2
            sorted_left, left_idx = merge_sort(arr, index_map, start, mid)
            sorted_right, right_idx = merge_sort(arr, index_map, mid, end)

            left, right = 0, 0
            merged_arr, merged_idx = [], []
            while left < len(sorted_left) and right < len(sorted_right):
                if sorted_left[left] < sorted_right[right]:
                    merged_arr.append(sorted_left[left])
                    merged_idx.append(left_idx[left])
                    left += 1
                else:
                    merged_arr.append(sorted_right[right])
                    merged_idx.append(right_idx[right])
                    right += 1
            merged_arr += sorted_left[left:]
            merged_idx += left_idx[left:]
            merged_arr += sorted_right[right:]
            merged_idx += right_idx[right:]
            return merged_arr, merged_idx

        arr = sorted(nums)
        index_map = {num: i for i, num in enumerate(nums)}
        bucket_map = defaultdict(list)

        for num in arr:
            bucket_key = num // (limit + 1)
            bucket_map[bucket_key].append(num)

        sorted_arr = []
        for bucket_key in sorted(bucket_map.keys()):
            sorted_in_bucket, _ = merge_sort(bucket_map[bucket_key], None, 0, len(bucket_map[bucket_key]))
            sorted_arr.extend(sorted_in_bucket)

        _, sorted_index = merge_sort(arr, index_map, 0, len(arr))
        result = [0] * len(nums)
        for i, idx in enumerate(sorted_index):
            result[idx] = sorted_arr[i]

        return result