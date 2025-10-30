def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    A = list(map(int, data[1:]))
    
    # Dictionary to store the indices of each color
    color_indices = {}
    
    for index, color in enumerate(A):
        if color in color_indices:
            color_indices[color].append(index)
        else:
            color_indices[color] = [index]
    
    # Count how many colors satisfy the condition
    count = 0
    for indices in color_indices.values():
        if abs(indices[0] - indices[1]) == 2:
            count += 1
    
    print(count)

if __name__ == "__main__":
    main()