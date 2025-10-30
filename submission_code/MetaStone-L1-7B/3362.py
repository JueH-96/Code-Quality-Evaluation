class Solution:
    def medianOfUniquenessArray(self, nums: List[int]) -> int:
        n = len(nums)
        L = n * (n + 1) // 2
        k = (L - 1) // 2

        def compute_f(x):
            count = 0
            left = 0
            freq = {}
            for right in range(n):
                num = nums[right]
                if num in freq:
                    if freq[num] == 1:
                        count -= 1
                    freq[num] += 1
                else:
                    freq[num] = 1
                while count > x:
                    left_num = nums[left]
                    if freq[left_num] == 1:
                        count -= 1
                    freq[left_num] -= 1
                    if freq[left_num] == 0:
                        del freq[left_num]
                    left += 1
                current_subarrays = right - left + 1
                count += current_subarrays
            return count

        low = 1
        high = n
        answer = n
        while low <= high:
            mid = (low + high) // 2
            f_mid = compute_f(mid)
            if f_mid >= k:
                answer = mid
                high = mid - 1
            else:
                low = mid + 1
        return answer