class Solution:
    spf = None
    max_val = 100000

    @classmethod
    def build_spf(cls, n):
        spf_arr = list(range(n+1))
        for i in range(2, int(n**0.5) + 1):
            if spf_arr[i] == i:
                for j in range(i*i, n+1, i):
                    if spf_arr[j] == j:
                        spf_arr[j] = i
        return spf_arr

    def __init__(self):
        if Solution.spf is None:
            Solution.spf = Solution.build_spf(Solution.max_val)

    def canTraverseAllPairs(self, nums):
        n = len(nums)
        if n == 1:
            return True
        if any(x == 1 for x in nums):
            return False
        
        parent = list(range(n))
        rank = [0] * n
        
        def find(x):
            root = x
            while root != parent[root]:
                root = parent[root]
            curr = x
            while curr != root:
                next_node = parent[curr]
                parent[curr] = root
                curr = next_node
            return root
        
        def union(x, y):
            rx = find(x)
            ry = find(y)
            if rx == ry:
                return
            if rank[rx] < rank[ry]:
                parent[rx] = ry
            elif rank[rx] > rank[ry]:
                parent[ry] = rx
            else:
                parent[ry] = rx
                rank[rx] += 1
        
        prime_first_occurrence = {}
        
        for i, x in enumerate(nums):
            temp = x
            factors = set()
            while temp > 1:
                p = Solution.spf[temp]
                factors.add(p)
                while temp % p == 0:
                    temp //= p
            for p in factors:
                if p in prime_first_occurrence:
                    union(i, prime_first_occurrence[p])
                else:
                    prime_first_occurrence[p] = i
        
        roots = set()
        for i in range(n):
            roots.add(find(i))
        return len(roots) == 1