class Solution:
    def minimumDistance(self, points: List[List[int]]) -> int:
        s_plus = [x + y for x, y in points]
        s_minus = [x - y for x, y in points]
        
        def compute_stats(arr):
            max1 = -float('inf')
            max2 = -float('inf')
            count_max1 = 0
            min1 = float('inf')
            min2 = float('inf')
            count_min1 = 0
            
            for num in arr:
                # Update max values
                if num > max1:
                    max2 = max1
                    max1 = num
                    count_max1 = 1
                elif num == max1:
                    count_max1 += 1
                elif num > max2:
                    max2 = num
                
                # Update min values
                if num < min1:
                    min2 = min1
                    min1 = num
                    count_min1 = 1
                elif num == min1:
                    count_min1 += 1
                elif num < min2:
                    min2 = num
            
            return {
                'max1': max1,
                'count_max1': count_max1,
                'max2': max2,
                'min1': min1,
                'count_min1': count_min1,
                'min2': min2,
            }
        
        stats_plus = compute_stats(s_plus)
        stats_minus = compute_stats(s_minus)
        
        min_candidate = float('inf')
        
        for i in range(len(points)):
            sp = s_plus[i]
            sm = s_minus[i]
            
            # Compute new_s_plus_max
            if sp == stats_plus['max1']:
                if stats_plus['count_max1'] > 1:
                    new_sp_max = stats_plus['max1']
                else:
                    new_sp_max = stats_plus['max2']
            else:
                new_sp_max = stats_plus['max1']
            
            # Compute new_s_plus_min
            if sp == stats_plus['min1']:
                if stats_plus['count_min1'] > 1:
                    new_sp_min = stats_plus['min1']
                else:
                    new_sp_min = stats_plus['min2']
            else:
                new_sp_min = stats_plus['min1']
            
            # Compute new_s_minus_max
            if sm == stats_minus['max1']:
                if stats_minus['count_max1'] > 1:
                    new_sm_max = stats_minus['max1']
                else:
                    new_sm_max = stats_minus['max2']
            else:
                new_sm_max = stats_minus['max1']
            
            # Compute new_s_minus_min
            if sm == stats_minus['min1']:
                if stats_minus['count_min1'] > 1:
                    new_sm_min = stats_minus['min1']
                else:
                    new_sm_min = stats_minus['min2']
            else:
                new_sm_min = stats_minus['min1']
            
            candidate_max = max(new_sp_max - new_sp_min, new_sm_max - new_sm_min)
            if candidate_max < min_candidate:
                min_candidate = candidate_max
        
        return min_candidate