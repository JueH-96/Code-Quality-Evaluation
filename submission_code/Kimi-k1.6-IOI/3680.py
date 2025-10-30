class Solution:
    def countComponents(self, nums: List[int], threshold: int) -> int:
        # Separate numbers into those <= threshold and others
        S = set()
        over = 0
        for num in nums:
            if num > threshold:
                over += 1
            else:
                S.add(num)
        
        if not S:
            return over
        
        # Create list of elements and mapping to indices
        elements = list(S)
        value_to_index = {val: idx for idx, val in enumerate(elements)}
        n = len(elements)
        
        # Initialize DSU
        parent = list(range(n))
        
        def find(u):
            while parent[u] != u:
                parent[u] = parent[parent[u]]  # Path compression
                u = parent[u]
            return u
        
        def union(u, v):
            pu = find(u)
            pv = find(v)
            if pu != pv:
                parent[pu] = pv  # Union without rank for simplicity
        
        # Precompute divisors for each m up to threshold using sieve method
        max_m = threshold
        divisors = [[] for _ in range(max_m + 1)]
        for d in range(1, max_m + 1):
            for m in range(d, max_m + 1, d):
                divisors[m].append(d)
        
        # Process each m from 1 to threshold
        for m in range(1, max_m + 1):
            # Collect all divisors of m that are in S
            D = []
            for d in divisors[m]:
                if d in value_to_index:
                    D.append(d)
            if len(D) < 1:
                continue
            # Union all elements in D together
            first_idx = value_to_index[D[0]]
            for num in D[1:]:
                current_idx = value_to_index[num]
                union(first_idx, current_idx)
        
        # Count the number of unique components in the DSU
        components = set()
        for i in range(n):
            components.add(find(i))
        
        return len(components) + over