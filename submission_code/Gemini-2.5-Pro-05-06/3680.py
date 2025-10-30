import math

class DSU:
    def __init__(self, n):
        self.parent = list(range(n))
        self.num_components = n

    def find(self, i):
        if self.parent[i] == i:
            return i
        self.parent[i] = self.find(self.parent[i])
        return self.parent[i]

    def union(self, i, j):
        root_i = self.find(i)
        root_j = self.find(j)
        if root_i != root_j:
            self.parent[root_i] = root_j
            self.num_components -= 1
            return True
        return False

class Solution:
  def countComponents(self, nums: list[int], threshold: int) -> int:
    n = len(nums)
    dsu = DSU(n)

    # Map values to their original indices in the nums array.
    # This helps quickly find the DSU index for a given number value.
    val_to_original_idx = {num: i for i, num in enumerate(nums)}

    # Iterate through all possible common divisor candidates `g`.
    # `g` can range from 1 up to `threshold`.
    # If g > threshold, then any multiples gx, gy would be > threshold,
    # and their lcm would also be > threshold.
    for g in range(1, threshold + 1):
        # Iterate for `x` such that `val_A = g*x`.
        # `val_A` must be `<= threshold`, so `x <= threshold / g`.
        # The loop `range(1, threshold // g + 1)` ensures `x` starts from 1
        # and `g*x` is at most `g * (threshold // g)`, which is `<= threshold`.
        for x in range(1, threshold // g + 1):
            val_A = g * x
            
            # If `val_A` is not in the input nums, skip.
            if val_A not in val_to_original_idx:
                continue
            
            idx_A = val_to_original_idx[val_A]

            # Iterate for `y` such that `val_B = g*y`.
            # `y` must be `> x` to consider distinct pairs (val_A, val_B) once.
            # `lcm(val_A, val_B) = g*x*y` if `gcd(x,y)=1`.
            # We need `g*x*y <= threshold`. So, `y <= threshold / (g*x)`.
            # `val_B = g*y` must also be `<= threshold`. So `y <= threshold / g`.
            # The condition `y <= threshold / (g*x)` is stricter as `x >= 1`.
            # Loop `range(x + 1, limit_y + 1)` ensures `g*y` is at most `g * (threshold // (g*x))`,
            # which is `<= threshold`.
            limit_y = threshold // (g * x)
            for y in range(x + 1, limit_y + 1):
                val_B = g * y

                # If `val_B` is not in the input nums, skip.
                if val_B not in val_to_original_idx:
                    continue
                
                # At this point:
                # 1. `val_A = g*x` is in `nums` (and `val_A <= threshold`).
                # 2. `val_B = g*y` is in `nums` (and `val_B <= threshold`).
                # 3. The construction of loops ensures `g*x*y <= threshold`.
                #
                # The condition for an edge between `val_A` and `val_B` is `lcm(val_A, val_B) <= threshold`.
                # If `gcd(x,y) == 1`, then `g` is `gcd(val_A, val_B)`.
                # In this case, `lcm(val_A, val_B) = (val_A * val_B) / g = (gx * gy) / g = g*x*y`.
                # Since `g*x*y <= threshold` is guaranteed by loop limits, the condition is met.
                if math.gcd(x,y) == 1:
                    idx_B = val_to_original_idx[val_B]
                    dsu.union(idx_A, idx_B)
    
    return dsu.num_components