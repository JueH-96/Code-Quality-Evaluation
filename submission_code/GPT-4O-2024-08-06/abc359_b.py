# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    A = list(map(int, data[1:]))
    
    # Dictionary to store the first occurrence of each color
    first_occurrence = {}
    
    # Counter for the number of colors that satisfy the condition
    count = 0
    
    for index, color in enumerate(A):
        if color in first_occurrence:
            # Check if there is exactly one person between the two occurrences
            if index - first_occurrence[color] == 2:
                count += 1
        else:
            # Store the first occurrence of the color
            first_occurrence[color] = index
    
    print(count)

if __name__ == "__main__":
    main()