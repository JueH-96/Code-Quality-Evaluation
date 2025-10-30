def count_triples(N, A):
    from collections import defaultdict

    # Dictionary to store the positions of each number
    positions = defaultdict(list)
    
    # Fill the positions dictionary
    for index in range(N):
        positions[A[index]].append(index + 1)  # Store 1-based index

    count = 0

    # Iterate through each unique number in the positions
    for num, pos_list in positions.items():
        m = len(pos_list)
        if m < 2:
            continue  # We need at least two occurrences of A_i = A_k

        # Calculate the number of valid (i, j, k) triples
        for j in range(1, m - 1):
            # pos_list[j] is A_j
            left_count = j  # Count of A_i before A_j
            right_count = m - (j + 1)  # Count of A_k after A_j
            
            # We need A_i != A_j, so we need to count how many A_j are not equal to A_i
            for other_num, other_pos_list in positions.items():
                if other_num != num:
                    count += left_count * right_count

    return count

import sys
input = sys.stdin.read
data = input().split()
N = int(data[0])
A = list(map(int, data[1:]))

result = count_triples(N, A)
print(result)