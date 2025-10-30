# YOUR CODE HERE
S = input().strip()
T = input().strip()

s_ptr = 0
t_ptr = 0
correct_indices = []

while s_ptr < len(S) and t_ptr < len(T):
    if S[s_ptr] == T[t_ptr]:
        correct_indices.append(t_ptr + 1)  # +1 because the problem uses 1-based indexing
        s_ptr += 1
        t_ptr += 1
    else:
        t_ptr += 1

print(' '.join(map(str, correct_indices)))