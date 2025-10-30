# YOUR CODE HERE
def find_correct_positions(s, t):
    positions = []
    s_index = 0
    for i, char in enumerate(t):
        if s_index < len(s) and char == s[s_index]:
            positions.append(i + 1)
            s_index += 1
    return positions

s = input().strip()
t = input().strip()

result = find_correct_positions(s, t)
print(' '.join(map(str, result)))