def count_fine_triplets(N, S):
    S = sorted(S)  # Sort the set of integers
    count = 0
    S_set = set(S)  # Use a set for O(1) average time complexity for lookups

    for i in range(N):
        for j in range(i + 1, N):
            A = S[i]
            B = S[j]
            C = 2 * B - A  # Calculate C based on A and B
            if C > B and C in S_set:  # Ensure C is greater than B and exists in the set
                count += 1

    return count

import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
S = list(map(int, data[1:N + 1]))

result = count_fine_triplets(N, S)
print(result)