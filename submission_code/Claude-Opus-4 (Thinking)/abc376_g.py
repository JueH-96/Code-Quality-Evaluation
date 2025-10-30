import heapq

def solve():
    MOD = 998244353
    
    t = int(input())
    
    for _ in range(t):
        n = int(input())
        parents = [0] + list(map(int, input().split()))
        a = [0] + list(map(int, input().split()))
        
        # Build adjacency list for children
        children = [[] for _ in range(n + 1)]
        for i in range(1, n + 1):
            children[parents[i]].append(i)
        
        # Calculate subtree sums
        subtree_sum = [0] * (n + 1)
        
        def dfs(v):
            subtree_sum[v] = a[v]
            for child in children[v]:
                subtree_sum[v] += dfs(child)
            return subtree_sum[v]
        
        dfs(0)
        
        # Simulate the greedy search
        searched = [False] * (n + 1)
        searched[0] = True
        d = [0] * (n + 1)
        operations = 0
        
        # Use a max heap (negate values for min heap)
        heap = []
        for child in children[0]:
            heapq.heappush(heap, (-subtree_sum[child], child))
        
        while heap:
            _, v = heapq.heappop(heap)
            if searched[v]:
                continue
            operations += 1
            searched[v] = True
            d[v] = operations
            
            for child in children[v]:
                if not searched[child]:
                    heapq.heappush(heap, (-subtree_sum[child], child))
        
        # Calculate expected value
        total_a = 0
        expected_numerator = 0
        for i in range(1, n + 1):
            total_a = (total_a + a[i]) % MOD
            expected_numerator = (expected_numerator + a[i] * d[i]) % MOD
        
        # Calculate expected_numerator / total_a mod MOD
        result = expected_numerator * pow(total_a, MOD - 2, MOD) % MOD
        
        print(result)

solve()