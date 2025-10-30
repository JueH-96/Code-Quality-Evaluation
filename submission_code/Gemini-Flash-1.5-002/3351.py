class Solution:
    def maximumHappinessSum(self, happiness: List[int], k: int) -> int:
        n = len(happiness)
        
        if k == 1:
            return max(happiness)
        
        ans = 0
        
        for i in range(1 << n):
            selected = []
            count = 0
            temp_happiness = list(happiness)
            current_sum = 0
            
            for j in range(n):
                if (i >> j) & 1:
                    selected.append(j)
                    count += 1

            if count != k:
                continue

            
            for idx in range(k):
                max_h = -1
                max_idx = -1
                for j in range(n):
                    if j not in selected[:idx]:
                        if temp_happiness[j] > max_h:
                            max_h = temp_happiness[j]
                            max_idx = j
                
                current_sum += max_h
                selected_idx = max_idx
                
                for j in range(n):
                    if j not in selected[:idx+1]:
                        temp_happiness[j] = max(0, temp_happiness[j]-1)

            ans = max(ans, current_sum)

        return ans