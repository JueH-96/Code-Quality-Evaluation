import sys

def solve():
    input = sys.stdin.read().split()
    ptr = 0
    T = int(input[ptr])
    ptr += 1
    for _ in range(T):
        N, K = map(int, input[ptr:ptr+2])
        ptr +=2
        cakes = []
        for __ in range(N):
            x, y, z = map(int, input[ptr:ptr+3])
            ptr +=3
            cakes.append((x, y, z))
        
        # Generate all 8 possible sign combinations
        patterns = [
            (1, 1, 1),
            (1, 1, -1),
            (1, -1, 1),
            (1, -1, -1),
            (-1, 1, 1),
            (-1, 1, -1),
            (-1, -1, 1),
            (-1, -1, -1)
        ]
        
        candidates = []
        for pattern in patterns:
            sx, sy, sz = pattern
            # Compute the key for each cake and sort
            temp = []
            for i in range(N):
                x, y, z = cakes[i]
                key = sx * x + sy * y + sz * z
                temp.append((key, i))
            # Sort in descending order
            temp.sort(reverse=True, key=lambda x: x[0])
            # Take top 2K indices
            top_indices = [idx for (key, idx) in temp[:2*K]]
            candidates.extend(top_indices)
        # Remove duplicates
        candidates = list(set(candidates))
        # Now, generate all possible pairs from these candidates
        m = len(candidates)
        pairs = []
        for i in range(m):
            for j in range(i+1, m):
                a = candidates[i]
                b = candidates[j]
                xa, ya, za = cakes[a]
                xb, yb, zb = cakes[b]
                price = max(xa + xb, ya + yb, za + zb)
                pairs.append((-price, a, b))  # using negative for min-heap
        # Now, we need to select K pairs with distinct elements and maximum total
        # But m could be up to 16K, so pairs could be 16K^2 which is 2.5e8 for K=5e4. Not feasible.
        # So, need a better approach.
        # Alternative: for each candidate, find the best possible partner not used yet.
        # So, sort all pairs by price, then use a greedy approach with a visited set.
        # But even sorting 2.5e8 elements is not feasible.
        # Hence, need another optimization.
        # So, instead, for each candidate, precompute the top K partners.
        # So, for each candidate in the list, collect all possible partners, compute the price, and keep top K.
        # Then, collect all these top pairs, sort them globally, and select the top K non-overlapping.
        
        # So, for each a in candidates, collect all other b in candidates, compute price, keep top 2K partners.
        # Then, the total pairs is O(16K * 2K) = 32K^2. For K=5e4, this is 8e9, which is still too much.
        # Hence, this approach is not feasible for large K.
        # So, need to find a smarter way.
        
        # Another idea: For each of the 8 patterns, the top pairs in those patterns are likely to include the optimal pairs.
        # So, for each pattern, generate the top K pairs, then merge all 8*K pairs, sort, and select top K non-overlapping.
        # Let's try this.
        all_possible_pairs = []
        for pattern in patterns:
            sx, sy, sz = pattern
            # Compute the key for each cake
            temp = []
            for i in range(N):
                x, y, z = cakes[i]
                key = sx * x + sy * y + sz * z
                temp.append((key, i))
            # Sort in descending order of key
            temp.sort(reverse=True, key=lambda x: x[0])
            # Take top 2K elements
            top_cakes = [idx for (key, idx) in temp[:2*K]]
            # Generate all pairs among these top 2K cakes
            current_pairs = []
            for i in range(len(top_cakes)):
                a = top_cakes[i]
                for j in range(i+1, len(top_cakes)):
                    b = top_cakes[j]
                    xa, ya, za = cakes[a]
                    xb, yb, zb = cakes[b]
                    price = max(xa + xb, ya + yb, za + zb)
                    current_pairs.append((-price, a, b))
            # Keep top K pairs from this pattern
            current_pairs_sorted = sorted(current_pairs)
            all_possible_pairs.extend(current_pairs_sorted[:2*K])  # Keep more to increase chances
        
        # Now, sort all possible pairs globally
        all_possible_pairs.sort()
        # Use a greedy approach to select K pairs with no overlapping elements
        used = set()
        res = 0
        count = 0
        for pair in all_possible_pairs:
            if count >= K:
                break
            price, a, b = pair
            price = -price
            if a not in used and b not in used:
                res += price
                used.add(a)
                used.add(b)
                count += 1
        print(res)

solve()