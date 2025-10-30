import sys

# Mapping for original moves
move_vec = {'N': (0, 1), 'S': (0, -1), 'E': (1, 0), 'W': (-1, 0)}
# Possible changes and their delta vectors (new_vec - orig_vec)
# Key: original char, Value: list of (new_char, delta_vec)
possible_changes_delta = {
    'N': [('S', (0, -2)), ('E', (1, -1)), ('W', (-1, -1))],
    'S': [('N', (0, 2)), ('E', (1, 1)), ('W', (-1, 1))],
    'E': [('N', (-1, 1)), ('S', (-1, -1)), ('W', (-2, 0))],
    'W': [('N', (1, 1)), ('S', (1, -1)), ('E', (2, 0))],
}

# Target sum functions
# p=0: x+y
# p=1: -x+y
# p=2: x-y
# p=3: -x-y
def apply_sum_func(x, y, p):
    if p == 0: return x + y
    elif p == 1: return -x + y
    elif p == 2: return x - y
    elif p == 3: return -x - y
    return 0 # Should not happen

def calculate_delta_p(delta_x, delta_y, p):
    if p == 0: return delta_x + delta_y
    elif p == 1: return -delta_x + delta_y
    elif p == 2: return delta_x - delta_y
    elif p == 3: return -delta_x - delta_y
    return 0

def check(s, k, D):
    n = len(s)

    # Calculate original prefix sums for the four target functions
    orig_f = [[0] * (n + 1) for _ in range(4)]
    x, y = 0, 0
    for i in range(n):
        dx, dy = move_vec[s[i]]
        x += dx
        y += dy
        for p in range(4):
            orig_f[p][i+1] = apply_sum_func(x, y, p)

    # Check maximizing f_p >= D for p = 0, 1, 2, 3
    for p in range(4):
        candidate_changes = [] # list of (benefit, index)
        for j in range(n):
            for _, (delta_x, delta_y) in possible_changes_delta[s[j]]:
                benefit = calculate_delta_p(delta_x, delta_y, p)
                if benefit > 0:
                    candidate_changes.append((benefit, j))

        # Sort candidate changes by benefit descending
        candidate_changes.sort(key=lambda item: item[0], reverse=True)

        # Greedily select up to k changes with positive benefit, ensuring unique indices
        selected_benefits_by_index = {} # Map index -> benefit
        for benefit, j in candidate_changes:
             if len(selected_benefits_by_index) == k: break
             if j not in selected_benefits_by_index:
                 selected_benefits_by_index[j] = benefit

        # Calculate the prefix sum of added benefits for the selected changes
        added_benefit_prefix_sum = [0] * (n + 1)
        for i in range(1, n + 1):
            added_benefit_prefix_sum[i] = added_benefit_prefix_sum[i-1]
            if (i-1) in selected_benefits_by_index:
                 added_benefit_prefix_sum[i] += selected_benefits_by_index[i-1]

        # Check if max f_p(P'_i) >= D
        for i in range(n + 1):
            if orig_f[p][i] + added_benefit_prefix_sum[i] >= D:
                return True

    # Check minimizing f_p <= -D for p = 0, 1, 2, 3
    for p in range(4):
        candidate_changes = [] # list of (abs(benefit), index) for negative benefits
        for j in range(n):
            for _, (delta_x, delta_y) in possible_changes_delta[s[j]]:
                benefit = calculate_delta_p(delta_x, delta_y, p)
                if benefit < 0:
                    candidate_changes.append((-benefit, j)) # Store positive absolute value

        # Sort candidate changes by absolute benefit descending
        candidate_changes.sort(key=lambda item: item[0], reverse=True)

        # Greedily select up to k changes with negative benefit, ensuring unique indices
        selected_benefits_by_index = {} # Map index -> abs(benefit)
        for abs_benefit, j in candidate_changes:
             if len(selected_benefits_by_index) == k: break
             if j not in selected_benefits_by_index:
                 selected_benefits_by_index[j] = abs_benefit

        # Calculate the prefix sum of added benefits for the selected changes
        added_benefit_prefix_sum = [0] * (n + 1)
        for i in range(1, n + 1):
            added_benefit_prefix_sum[i] = added_benefit_prefix_sum[i-1]
            if (i-1) in selected_benefits_by_index:
                # The actual benefit is negative
                benefit = -selected_benefits_by_index[i-1]
                added_benefit_prefix_sum[i] += benefit

        # Check if min f_p(P'_i) <= -D
        for i in range(n + 1):
            if orig_f[p][i] + added_benefit_prefix_sum[i] <= -D:
                return True

    return False # Cannot achieve distance D

class Solution:
    def maxDistance(self, s: str, k: int) -> int:
        # Binary search for the maximum distance D
        # The distance is Manhattan distance |x| + |y| = max(|x+y|, |-x+y|, |x-y|, |-x-y|)
        # So we are searching for max D such that check(D) is True.

        # Upper bound for Manhattan distance:
        # Maximum coordinate can be roughly N (straight line) + k * max_coord_change (2)
        # Max |x| or |y| <= N + 2k.
        # Max |x| + |y| <= |N+2k| + |N+2k| = 2N + 4k.
        # For N=10^5, k=10^5, max dist <= 2e5 + 4e5 = 6e5.
        # A safe binary search upper bound could be 2 * len(s) + 4 * k + 5.
        high = 2 * len(s) + 4 * k + 5
        low = 0
        ans = 0

        while low <= high:
            mid = (low + high) // 2
            if check(s, k, mid):
                ans = mid
                low = mid + 1
            else:
                high = mid - 1

        return ans