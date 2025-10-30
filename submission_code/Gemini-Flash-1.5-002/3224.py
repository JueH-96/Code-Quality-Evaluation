class Solution:
    def numberOfSequence(self, n: int, sick: List[int]) -> int:
        mod = 10**9 + 7
        uninfected = []
        for i in range(n):
            if i not in sick:
                uninfected.append(i)
        m = len(uninfected)
        if m == 0:
            return 1
        dp = {}
        def count_sequences(idx, left, right):
            if idx == m:
                return 1
            if (idx, left, right) in dp:
                return dp[(idx, left, right)]
            count = 0
            curr = uninfected[idx]
            if left != -1 and abs(curr - left) == 1:
                count = (count + count_sequences(idx + 1, curr, right)) % mod
            if right != -1 and abs(curr - right) == 1:
                count = (count + count_sequences(idx + 1, left, curr)) % mod
            
            dp[(idx, left, right)] = count
            return count

        
        ans = 0
        
        
        first_uninfected_indices = []
        for i in range(m):
            
            is_first = True
            curr = uninfected[i]
            
            left_neighbor_infected = False
            if curr > 0:
                if curr -1 in sick:
                    left_neighbor_infected = True
            
            right_neighbor_infected = False
            if curr < n -1:
                if curr + 1 in sick:
                    right_neighbor_infected = True
            
            if not left_neighbor_infected and not right_neighbor_infected:
                is_first = False
            
            if is_first:
                first_uninfected_indices.append(i)

        for start_idx in first_uninfected_indices:
            ans = (ans + count_sequences(start_idx +1, uninfected[start_idx], -1 if start_idx + 1 >= m else uninfected[start_idx+1] )) % mod
            ans = (ans + count_sequences(start_idx +1, -1 if start_idx -1 < 0 else uninfected[start_idx-1], uninfected[start_idx])) % mod

        return ans