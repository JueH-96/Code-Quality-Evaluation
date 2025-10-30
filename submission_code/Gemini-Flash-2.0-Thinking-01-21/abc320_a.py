input_line = input()
input_str_list = input_line.split()
A = int(input_str_list[0])
B = int(input_str_list[1])
result = A**B + B**A
print(result)