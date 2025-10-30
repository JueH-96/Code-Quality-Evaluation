def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    S = list(data[2])
    C = list(map(int, data[3:]))
    
    # Create a dictionary to hold positions of each color
    positions = {i: [] for i in range(1, M + 1)}
    
    # Fill the dictionary with positions for each color
    for index, color in enumerate(C):
        positions[color].append(index)
    
    # Create a new list to hold the transformed string
    new_S = [''] * N
    
    # Perform the right circular shift for each color
    for color in range(1, M + 1):
        pos_list = positions[color]
        if not pos_list:
            continue
        # Number of positions for this color
        k = len(pos_list)
        # Right circular shift by 1
        for i in range(k):
            new_S[pos_list[i]] = S[pos_list[(i - 1) % k]]
    
    # Print the final transformed string
    print(''.join(new_S))

if __name__ == "__main__":
    main()