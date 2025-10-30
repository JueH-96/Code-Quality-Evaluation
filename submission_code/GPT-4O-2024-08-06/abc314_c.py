# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    S = list(data[2])
    C = list(map(int, data[3:]))
    
    # Dictionary to hold indices for each color
    color_indices = {i: [] for i in range(1, M + 1)}
    
    # Fill the dictionary with indices for each color
    for index, color in enumerate(C):
        color_indices[color].append(index)
    
    # Perform the right circular shift for each color
    for color in range(1, M + 1):
        indices = color_indices[color]
        if len(indices) > 1:
            # Get the last character for the right circular shift
            last_char = S[indices[-1]]
            # Shift all characters to the right
            for i in range(len(indices) - 1, 0, -1):
                S[indices[i]] = S[indices[i - 1]]
            # Place the last character at the first position
            S[indices[0]] = last_char
    
    # Join the list back into a string and print the result
    print(''.join(S))