S = input()
first_pipe = S.find('|')
second_pipe = S.rfind('|')
result = S[0:first_pipe] + S[second_pipe+1:]
print(result)