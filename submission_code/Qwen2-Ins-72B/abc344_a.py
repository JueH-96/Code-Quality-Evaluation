S = input()
first_bar_index = S.find('|')
second_bar_index = S.rfind('|')

result = S[:first_bar_index] + S[second_bar_index+1:]
print(result)