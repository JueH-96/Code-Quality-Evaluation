def solve():
    s = input()
    position_map = {}
    for i in range(len(s)):
        position_map[s[i]] = i + 1
    
    total_distance = 0
    previous_char = 'A'
    
    for current_char_code in range(ord('B'), ord('Z') + 1):
        current_char = chr(current_char_code)
        distance = abs(position_map[current_char] - position_map[previous_char])
        total_distance += distance
        previous_char = current_char
        
    print(total_distance)

if __name__ == '__main__':
    solve()