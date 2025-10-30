s = input()
first_pipe = s.find('|')
second_pipe = s.find('|', first_pipe + 1)
result = s[:first_pipe] + s[second_pipe + 1:]
print(result)