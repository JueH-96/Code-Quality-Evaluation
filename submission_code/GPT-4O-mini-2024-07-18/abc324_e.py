def count_pairs(N, T, strings):
    from collections import defaultdict

    # Precompute the positions of each character in T
    char_positions = defaultdict(list)
    for index, char in enumerate(T):
        char_positions[char].append(index)

    # Function to find the maximum index of T that can be formed as a subsequence
    def max_t_index(s):
        t_index = -1
        for char in s:
            if char in char_positions:
                for pos in char_positions[char]:
                    if pos > t_index:
                        t_index = pos
                        if t_index == len(T) - 1:
                            return t_index
        return t_index

    # Count how many strings can contribute to each position in T
    prefix_count = [0] * (len(T) + 1)
    for s in strings:
        max_index = max_t_index(s)
        if max_index >= 0:
            prefix_count[max_index + 1] += 1

    # Calculate the number of valid pairs
    total_pairs = 0
    suffix_count = 0
    for i in range(len(T), 0, -1):
        suffix_count += prefix_count[i]
        total_pairs += prefix_count[i - 1] * suffix_count

    return total_pairs

import sys
input = sys.stdin.read
data = input().splitlines()

N, T = data[0].split()
N = int(N)
strings = data[1:N + 1]

result = count_pairs(N, T, strings)
print(result)