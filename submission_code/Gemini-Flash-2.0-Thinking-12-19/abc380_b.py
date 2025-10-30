S = input()
pipe_indices = []
for i in range(len(S)):
    if S[i] == '|':
        pipe_indices.append(i)
A_sequence = []
for i in range(1, len(pipe_indices)):
    A_i = pipe_indices[i] - pipe_indices[i-1] - 1
    A_sequence.append(A_i)
result_str = ""
for i in range(len(A_sequence)):
    result_str += str(A_sequence[i])
    if i < len(A_sequence) - 1:
        result_str += " "
print(result_str)