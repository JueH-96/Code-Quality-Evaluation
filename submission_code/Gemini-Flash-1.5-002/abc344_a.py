s = input()
first_pipe = s.find('|')
second_pipe = s.rfind('|')
print(s[:first_pipe] + s[second_pipe+1:])