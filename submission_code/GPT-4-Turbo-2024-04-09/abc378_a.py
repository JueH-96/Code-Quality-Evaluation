import sys
input = sys.stdin.read

def main():
    # Read input
    data = input().strip()
    # Split the input into a list of integers
    colors = list(map(int, data.split()))
    
    # Count the frequency of each color
    color_count = {}
    for color in colors:
        if color in color_count:
            color_count[color] += 1
        else:
            color_count[color] = 1
    
    # Calculate the number of pairs we can form
    pairs = 0
    for count in color_count.values():
        pairs += count // 2
    
    # Print the result
    print(pairs)

if __name__ == "__main__":
    main()