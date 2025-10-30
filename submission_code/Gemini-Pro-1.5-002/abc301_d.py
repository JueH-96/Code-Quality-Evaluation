# YOUR CODE HERE
s = input()
n = int(input())

q_count = s.count('?')

ans = -1

for i in range(2**q_count):
    temp = bin(i)[2:].zfill(q_count)
    current_num_str = ""
    temp_index = 0
    for char in s:
        if char == '?':
            current_num_str += temp[temp_index]
            temp_index += 1
        else:
            current_num_str += char
    
    current_num = int(current_num_str, 2)
    
    if current_num <= n:
        ans = max(ans, current_num)

print(ans)