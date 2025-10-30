def main():
    import sys
    sys.setrecursionlimit(1 << 25)
    N, M = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))
    
    # Step 1: Compute prefix_sum[r] = (prefix_sum[r-1] + A[r]) % M
    prefix_sum = [0] * (N + 1)
    for r in range(1, N + 1):
        prefix_sum[r] = (prefix_sum[r - 1] + A[r - 1]) % M
    
    # Step 2: Compute sum_total
    sum_total_part1 = 0
    for r in range(1, N + 1):
        sum_total_part1 += prefix_sum[r] * r
    sum_total_part2 = 0
    for l in range(1, N + 1):
        sum_total_part2 += prefix_sum[l - 1] * (N - l + 1)
    sum_total = sum_total_part1 - sum_total_part2
    
    # Step 3: Compute sum_floor using Fenwick Tree
    class FenwickTree:
        def __init__(self, size):
            self.N = size
            self.tree = [0] * (self.N + 2)
        
        def update(self, index, value=1):
            while index <= self.N:
                self.tree[index] += value
                index += index & -index
        
        def query(self, index):
            res = 0
            while index > 0:
                res += self.tree[index]
                index -= index & -index
            return res
    
    # Sort unique prefix_sum values for coordinate compression
    unique_prefix = sorted(set(prefix_sum))
    mapping = {v: i+1 for i, v in enumerate(unique_prefix)}
    
    ft = FenwickTree(len(unique_prefix))
    sum_floor = 0
    for r in range(1, N + 1):
        target = prefix_sum[r] - M
        if target >= 0:
            idx = bisect.bisect_left(unique_prefix, target)
            if idx < len(unique_prefix):
                sum_floor += ft.query(len(unique_prefix)) - ft.query(idx)
        ft.update(mapping[prefix_sum[r-1]])
    
    # Final answer
    answer = sum_total - M * sum_floor
    print(answer)

if __name__ == "__main__":
    main()