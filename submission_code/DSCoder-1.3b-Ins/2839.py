class Solution:
    def maximumSumQueries(self, nums1: List[int], nums2: List[int], queries: List[List[int]]) -> List[int]:
        n = len(nums1)
        arr = sorted(nums1)
        arr2 = sorted(nums2)
        prefix = [0]*(n+1)
        suffix = [0]*(n+1)
        for i in range(n-1, -1, -1):
            suffix[i] = arr[i] + suffix[i+1]
        result = []
        for q in queries:
            x, y = q
            l1, r1, l2, r2 = 0, n, 0, n
            while l1 < r1:
                mid1 = l1 + (r1 - l1) // 2
                mid2 = r2 - (r2 - r1) // 2
                if arr[mid1] + arr2[mid2] >= x:
                    r1 = mid1
                else:
                    l1 = mid1 + 1
            while l2 < r2:
                mid1 = l2 + (r2 - l2) // 2
                mid2 = r1 - (r1 - r2) // 2
                if arr[mid1] + arr2[mid2] >= y:
                    r2 = mid1
                else:
                    l2 = mid1 + 1
            if l1 == r1 and l2 == r2:
                result.append(-1)
            else:
                result.append(max(arr[l1] + suffix[r1], arr[l2] + suffix[r2]))
        return result