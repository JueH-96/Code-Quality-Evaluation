class Solution:
    def countOfPeaks(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        def count_peaks(arr, left, right):
            if right - left < 2:
                return 0
            count = 0
            for i in range(left + 1, right):
                if arr[i] > arr[i-1] and arr[i] > arr[i+1]:
                    count += 1
            return count
        
        answer = []
        for query in queries:
            if query[0] == 1:
                # Count peaks in subarray
                answer.append(count_peaks(nums, query[1], query[2]))
            else:
                # Update value
                nums[query[1]] = query[2]
                
        return answer