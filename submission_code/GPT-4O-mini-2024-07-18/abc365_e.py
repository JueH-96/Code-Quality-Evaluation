def calculate_xor_sum(N, A):
    total_sum = 0
    prefix_xor = [0] * (N + 1)

    # Calculate prefix XOR
    for i in range(1, N + 1):
        prefix_xor[i] = prefix_xor[i - 1] ^ A[i - 1]

    # Calculate the total sum of XORs for all pairs (i, j)
    for i in range(1, N):
        for j in range(i + 1, N + 1):
            total_sum += prefix_xor[j] ^ prefix_xor[i - 1]

    return total_sum

import sys
input = sys.stdin.read

data = input().split()
N = int(data[0])
A = list(map(int, data[1:]))

result = calculate_xor_sum(N, A)
print(result)