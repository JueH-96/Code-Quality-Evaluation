class Solution:
    def minOperations(self, queries: List[List[int]]) -> int:
        # Precompute powers of 4 up to 10^9
        powers_of_4 = [1]
        while powers_of_4[-1] <= 10**9:
            powers_of_4.append(powers_of_4[-1] * 4)
        
        def compute_total_work(l, r):
            total_work = 0
            for k in range(1, len(powers_of_4)):
                start = powers_of_4[k-1]
                end = powers_of_4[k] - 1
                if start > r:
                    break
                intersection_start = max(l, start)
                intersection_end = min(r, end)
                if intersection_start <= intersection_end:
                    count = intersection_end - intersection_start + 1
                    total_work += k * count
            return total_work
        
        result = 0
        for l, r in queries:
            total_work = compute_total_work(l, r)
            operations = (total_work + 1) // 2  # ceil(total_work / 2)
            result += operations
        
        return result