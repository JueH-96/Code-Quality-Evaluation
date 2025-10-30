s = input()
pos = {}
for i, char in enumerate(s):
  pos[char] = i + 1

total_distance = 0
current_pos = pos['A']
for char_code in range(ord('B'), ord('Z') + 1):
  next_char = chr(char_code)
  next_pos = pos[next_char]
  total_distance += abs(next_pos - current_pos)
  current_pos = next_pos

print(total_distance)