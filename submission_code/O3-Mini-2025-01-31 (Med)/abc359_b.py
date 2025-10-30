def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    N = int(data[0])
    A = list(map(int, data[1:]))
    # Create dictionary to store the first and second occurrence of each color.
    positions = {}
    for idx, color in enumerate(A):
        if color in positions:
            positions[color].append(idx)
        else:
            positions[color] = [idx]
    
    count = 0
    # For each color, check if the two people's positions differ by 2 (i.e. exactly one person in between)
    for color, pos_list in positions.items():
        if pos_list[1] - pos_list[0] == 2:
            count += 1
            
    sys.stdout.write(str(count))
    
if __name__ == "__main__":
    main()