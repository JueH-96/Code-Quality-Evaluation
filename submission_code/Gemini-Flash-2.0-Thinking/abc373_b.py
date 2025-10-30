s = input()
current_position = s.index('A')
total_distance = 0
for char_code in range(ord('B'), ord('Z') + 1):
    char = chr(char_code)
    target_position = s.index(char)
    distance = abs(target_position - current_position)
    total_distance += distance
    current_position = target_position
print(total_distance)