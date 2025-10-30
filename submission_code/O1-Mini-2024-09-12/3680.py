class Solution:
    def countComponents(self, nums: List[int], threshold: int) -> int:
        # Filter numbers <= threshold
        relevant_nums = [x for x in nums if x <= threshold]
        m = len(relevant_nums)
        k = len(nums) - m  # Numbers > threshold are individual components
        
        if m == 0:
            return k
        
        # Map number to index
        num_to_index = {num: idx for idx, num in enumerate(relevant_nums)}
        
        # Initialize Union-Find
        parent = list(range(m))
        rank = [1] * m
        
        def find(u):
            while parent[u] != u:
                parent[u] = parent[parent[u]]
                u = parent[u]
            return u
        
        def union(u, v):
            pu, pv = find(u), find(v)
            if pu == pv:
                return
            if rank[pu] < rank[pv]:
                parent[pu] = pv
            else:
                parent[pv] = pu
                if rank[pu] == rank[pv]:
                    rank[pu] += 1
        
        # Create presence array
        presence = [False] * (threshold + 1)
        for num in relevant_nums:
            presence[num] = True
        
        # Iterate through all possible divisors
        for k_div in range(1, threshold + 1):
            first_num = -1
            for multiple in range(k_div, threshold + 1, k_div):
                if presence[multiple]:
                    if first_num == -1:
                        first_num = multiple
                    else:
                        union(num_to_index[first_num], num_to_index[multiple])
        
        # Count unique parents
        unique_parents = set()
        for i in range(m):
            unique_parents.add(find(i))
        
        return len(unique_parents) + k