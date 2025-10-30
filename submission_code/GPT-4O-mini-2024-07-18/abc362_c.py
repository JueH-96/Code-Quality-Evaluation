def find_sequence(N, pairs):
    total_min = sum(L for L, R in pairs)
    total_max = sum(R for L, R in pairs)

    if total_min > 0 or total_max < 0:
        return "No"

    # Start with the minimum values
    X = [L for L, R in pairs]
    current_sum = total_min

    # We need to adjust the sum to be zero
    for i in range(N):
        L, R = pairs[i]
        # The maximum we can add to X[i] is R - L
        max_increase = R - L
        # We need to increase current_sum to reach 0
        needed_increase = -current_sum
        
        if needed_increase > 0:
            increase = min(max_increase, needed_increase)
            X[i] += increase
            current_sum += increase
        
        if current_sum == 0:
            break

    if current_sum == 0:
        return "Yes
" + " ".join(map(str, X))
    else:
        return "No"

import sys
input = sys.stdin.read

data = input().strip().splitlines()
N = int(data[0])
pairs = [tuple(map(int, line.split())) for line in data[1:]]

result = find_sequence(N, pairs)
print(result)