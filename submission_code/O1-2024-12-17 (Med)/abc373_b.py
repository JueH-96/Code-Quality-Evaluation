def main():
    import sys
    
    S = sys.stdin.readline().strip()
    # Create a dictionary to store the position of each character
    position = {}
    for i, char in enumerate(S):
        position[char] = i + 1
    
    # We'll type the letters A to Z in order
    from string import ascii_uppercase
    letters = ascii_uppercase
    
    # Start from the position of 'A'
    total_distance = 0
    prev_pos = position['A']
    
    # Calculate the distance for subsequent letters
    for char in letters[1:]:
        curr_pos = position[char]
        total_distance += abs(curr_pos - prev_pos)
        prev_pos = curr_pos
    
    print(total_distance)

# Do not forget to call main
main()