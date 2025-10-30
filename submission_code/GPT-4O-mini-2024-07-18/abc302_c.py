def can_rearrange_to_sequence(N, M, strings):
    from itertools import permutations
    
    def can_transform(s1, s2):
        # Check if we can change exactly one character in s1 to get s2
        diff_count = sum(1 for a, b in zip(s1, s2) if a != b)
        return diff_count == 1
    
    for perm in permutations(strings):
        valid_sequence = True
        for i in range(N - 1):
            if not can_transform(perm[i], perm[i + 1]):
                valid_sequence = False
                break
        if valid_sequence:
            return "Yes"
    
    return "No"

import sys
input = sys.stdin.read
data = input().splitlines()

N, M = map(int, data[0].split())
strings = data[1:N + 1]

result = can_rearrange_to_sequence(N, M, strings)
print(result)