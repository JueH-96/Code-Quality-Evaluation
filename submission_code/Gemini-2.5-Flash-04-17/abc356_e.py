import sys

class FenwickTree:
    def __init__(self, max_value):
        self.max_value = max_value
        self.tree = [0] * (max_value + 1)

    def add(self, value, count):
        # value is 1-based
        index = value
        while index <= self.max_value:
            self.tree[index] += count
            index += index & (-index)

    def query_prefix(self, value):
        # value is 1-based. Query sum up to value.
        index = value
        sum_val = 0
        while index > 0:
            sum_val += self.tree[index]
            index -= index & (-index)
        return sum_val

    def query_range(self, l, r):
        # l, r are 1-based values
        if l > r:
            return 0
        return self.query_prefix(r) - self.query_prefix(l - 1)

def solve():
    N = int(sys.stdin.readline())
    A = list(map(int, sys.stdin.readline().split()))

    MAX_VAL = 10**6
    bit_counts = FenwickTree(MAX_VAL) # Stores prefix counts for 1-based values

    total_sum = 0

    # Iterate through the array from left to right (processing A[i])
    # When processing A[i], consider pairs (A[k], A[i]) where k < i.
    # The BIT stores counts of values A[k] for k < i.
    for i in range(N):
        v = A[i] # A[i] is 1-based value

        # Calculate sum for pairs (A[k], A[i]) where k < i.
        # Contribution is floor(max(A[k], v) / min(A[k], v)).
        # Split into A[k] = x <= v and A[k] = x > v.

        # Case 1: A[k] = x <= v. Term = floor(v/x).
        # Sum = sum_{x=1 to v} count(A[k]=x, k<i) * floor(v/x).
        # Use number theoretic block decomposition (sqrt decomposition) for sum over x.
        sqrt_v = int(v**0.5)
        
        # Part 1.1: x from 1 to sqrt(v). Direct summation.
        # We need count(A[k]=x, k<i) for each x. This is query_range(x, x).
        for x in range(1, sqrt_v + 1):
            # x is always >= 1. Check if x <= MAX_VAL (always true since x <= sqrt(v) <= sqrt(MAX_VAL))
            count_x = bit_counts.query_range(x, x)
            if count_x > 0:
                total_sum += count_x * (v // x)

        # Part 1.2: x from sqrt(v)+1 to v. Group by floor(v/x) = k.
        # For x in [sqrt_v + 1, v], floor(v/x) ranges from floor(v/v)=1 up to floor(v/(sqrt_v + 1)).
        # k = floor(v/x) is constant for x in (v/(k+1), v/k].
        # We need x in [sqrt_v + 1, v] intersecting with (v/(k+1), v/k].
        
        # Max k value in this part is floor(v / (sqrt_v + 1)) if sqrt_v + 1 <= v else 0
        max_k_part1_2 = 0
        if sqrt_v + 1 <= v:
             max_k_part1_2 = v // (sqrt_v + 1)
        
        # Iterate k from 1 up to max_k_part1_2
        for k in range(1, max_k_part1_2 + 1):
            # x is in (v/(k+1), v/k]
            lower_bound_x_for_k = v // (k + 1) + 1
            upper_bound_x_for_k = v // k
            
            # We need x in [sqrt_v + 1, v] intersecting with [lower_bound_x_for_k, upper_bound_x_for_k]
            range_l = max(sqrt_v + 1, lower_bound_x_for_k)
            range_r = min(v, upper_bound_x_for_k)

            # Ensure range_l and range_r are within valid value bounds [1, MAX_VAL]
            range_l = max(1, range_l)
            range_r = min(MAX_VAL, range_r)

            if range_l <= range_r:
                count_range = bit_counts.query_range(range_l, range_r)
                if count_range > 0:
                    total_sum += count_range * k

        # Case 2: A[k] = x > v. Term = floor(x/v).
        # Sum is sum_{x=v+1 to MAX_VAL} count(A[k]=x, k<i) * floor(x/v)
        # Group by floor(x/v) = k.
        # For x in [v+1, MAX_VAL], floor(x/v) ranges from floor((v+1)/v) up to floor(MAX_VAL/v).
        # k = floor(x/v) is constant for x in [kv, (k+1)v-1].
        # We need x in [v+1, MAX_VAL] intersecting with [kv, (k+1)v-1].
        # Range of x: [max(v+1, kv), min(MAX_VAL, (k+1)v-1)]

        if v < MAX_VAL: # Only sum if there are values > v
            # k ranges from floor((v+1)/v) up to floor(MAX_VAL/v).
            # Smallest possible k is floor((v+1)/v). Since v >= 1, v+1 > v, so (v+1)/v > 1. floor((v+1)/v) >= 1.
            # The loop `range(1, MAX_VAL // v + 1)` covers k = 1, 2, ..., floor(MAX_VAL/v).
            # For k < floor((v+1)/v), the range [kv, (k+1)v-1] starts at kv.
            # If k=1 and v=1, [1, 1]. Intersection with [2, MAX_VAL] is empty.
            # If k=1 and v>1, [v, 2v-1]. Intersection with [v+1, MAX_VAL] is [v+1, min(2v-1, MAX_VAL)].
            # The lower bound calculation `max(v + 1, lower_bound_x_for_k)` correctly handles these cases.
            
            for k in range(1, MAX_VAL // v + 1):
                lower_bound_x_for_k = k * v
                upper_bound_x_for_k = (k + 1) * v - 1
                
                # We need x in [v+1, MAX_VAL] intersecting with [lower_bound_x_for_k, upper_bound_x_for_k]
                range_l = max(v + 1, lower_bound_x_for_k)
                range_r = min(MAX_VAL, upper_bound_x_for_k)
                
                # Ensure range_l and range_r are within valid value bounds [1, MAX_VAL]
                range_l = max(1, range_l)
                range_r = min(MAX_VAL, range_r)

                if range_l <= range_r:
                    count_range = bit_counts.query_range(range_l, range_r)
                    if count_range > 0:
                         total_sum += count_range * k

        # Add A[i] to the BIT for future iterations
        bit_counts.add(v, 1)

    print(total_sum)

solve()