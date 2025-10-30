# Read the input
N, M = map(int, input().split())
S = input()
T = input()

# Convert the strings to lists of integers
S_list = [int(c) for c in S]
T_list = [int(c) for c in T]

# Perform the operations
for k in range(1, M+1):
    max_idx = max(range(N), key=lambda i: S_list[i] if i < N else 0)
    S_list[max_idx] = T_list[k-1]

# Convert the list back to a string and print the result
print(''.join(map(str, S_list)))