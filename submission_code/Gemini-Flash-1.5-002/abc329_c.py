# YOUR CODE HERE
n = int(input())
s = input()
count = 0
current_char = ''
current_count = 0
for char in s:
    if char == current_char:
        current_count += 1
    else:
        if current_count > 0:
            count += current_count * (current_count + 1) // 2
        current_char = char
        current_count = 1
if current_count > 0:
    count += current_count * (current_count + 1) // 2
print(count)