s = input()
first_pipe_index = s.find('|')
second_pipe_index = s.rfind('|')
result = s[:first_pipe_index] + s[second_pipe_index + 1:]
print(result)