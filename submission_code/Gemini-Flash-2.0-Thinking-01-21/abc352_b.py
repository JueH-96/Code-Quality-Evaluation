s = input()
t = input()
s_len = len(s)
t_len = len(t)
correct_positions = []
s_index = 0
t_index = 0
while s_index < s_len:
    target_char = s[s_index]
    while t_index < t_len:
        current_char = t[t_index]
        if current_char == target_char:
            correct_positions.append(t_index + 1)
            s_index += 1
            t_index += 1
            break
        else:
            t_index += 1
    else:
        # This should not happen based on problem description, but for robustness, maybe handle it.
        # If we reach end of T without finding a match for s[s_index], it's an error case.
        # However, problem guarantees that T is formed by the given process, so we should always find a match.
        break

output_str = ""
for i in range(len(correct_positions)):
    output_str += str(correct_positions[i])
    if i < len(correct_positions) - 1:
        output_str += " "

print(output_str)