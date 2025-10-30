# YOUR CODE HERE
def is_possible_t(T_prime, S):
    len_T_prime = len(T_prime)
    len_S = len(S)
    
    if len_T_prime == len_S:
        # Check if T' is equal to S or if T' can be obtained by changing one character in S
        diff_count = sum(1 for a, b in zip(T_prime, S) if a != b)
        return diff_count == 0 or diff_count == 1
    elif len_T_prime == len_S + 1:
        # Check if T' can be obtained by inserting one character in S
        for i in range(len_T_prime):
            if T_prime[:i] + T_prime[i+1:] == S:
                return True
        return False
    elif len_T_prime + 1 == len_S:
        # Check if T' can be obtained by deleting one character from S
        for i in range(len_S):
            if S[:i] + S[i+1:] == T_prime:
                return True
        return False
    else:
        return False

import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
T_prime = data[1]
S_list = data[2:]

result_indices = []

for i in range(N):
    if is_possible_t(T_prime, S_list[i]):
        result_indices.append(i + 1)

print(len(result_indices))
if result_indices:
    print(" ".join(map(str, result_indices)))