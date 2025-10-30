S = input()
first_pipe = S.find('|')
second_pipe = S.find('|', first_pipe + 1)
result = S[:first_pipe] + S[second_pipe + 1:]
print(result)