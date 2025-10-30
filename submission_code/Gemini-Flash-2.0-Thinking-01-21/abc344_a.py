s = input()
pipe_indices = []
for i in range(len(s)):
    if s[i] == '|':
        pipe_indices.append(i)

first_pipe_index = pipe_indices[0]
second_pipe_index = pipe_indices[1]

result = s[:first_pipe_index] + s[second_pipe_index+1:]
print(result)