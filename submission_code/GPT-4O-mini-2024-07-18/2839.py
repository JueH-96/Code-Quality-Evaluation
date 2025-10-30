class Solution:
    def maximumSumQueries(self, nums1: List[int], nums2: List[int], queries: List[List[int]]) -> List[int]:
        from sortedcontainers import SortedList
        
        n = len(nums1)
        combined = [(nums1[i] + nums2[i], nums1[i], nums2[i]) for i in range(n)]
        combined.sort(reverse=True, key=lambda x: x[0])  # Sort by the sum nums1[j] + nums2[j] in descending order
        
        results = []
        sorted_nums1 = SortedList()
        sorted_nums2 = SortedList()
        
        for x, y in queries:
            max_sum = -1
            
            # Add elements to the sorted lists that meet the criteria
            for value, num1, num2 in combined:
                if num1 >= x and num2 >= y:
                    sorted_nums1.add(num1)
                    sorted_nums2.add(num2)
                    max_sum = max(max_sum, value)
                elif num1 < x and num2 < y:
                    break  # No need to check further as the list is sorted
            
            results.append(max_sum)
        
        return results