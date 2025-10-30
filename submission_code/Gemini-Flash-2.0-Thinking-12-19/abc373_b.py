def solve():
    s = input()
    char_positions = {}
    for i in range(len(s)):
        char_positions[s[i]] = i + 1
    
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    total_distance = 0
    
    for i in range(len(alphabet) - 1):
        char1 = alphabet[i]
        char2 = alphabet[i+1]
        pos1 = char_positions[char1]
        pos2 = char_positions[char2]
        distance = abs(pos2 - pos1)
        total_distance += distance
        
    print(total_distance)

if __name__ == '__main__':
    solve()