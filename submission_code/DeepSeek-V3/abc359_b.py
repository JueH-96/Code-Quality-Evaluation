# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    A = list(map(int, data[1:2*N+1]))
    
    color_to_indices = {}
    for idx, color in enumerate(A):
        if color not in color_to_indices:
            color_to_indices[color] = []
        color_to_indices[color].append(idx + 1)  # +1 to convert to 1-based index
    
    count = 0
    for color in color_to_indices:
        pos1, pos2 = color_to_indices[color]
        if abs(pos1 - pos2) == 2:
            count += 1
    
    print(count)

if __name__ == "__main__":
    main()