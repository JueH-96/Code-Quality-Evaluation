def main():
    N = int(input())
    horizontal_strings = [input() for _ in range(N)]
    
    # Find the maximum length of the horizontal strings
    M = max(len(s) for s in horizontal_strings)
    
    # Initialize M vertical strings as lists, each with N positions
    vertical_strings = [['*'] * N for _ in range(M)]
    
    # Populate the vertical strings
    for i, s in enumerate(horizontal_strings):
        position = N - i - 1
        for j, char in enumerate(s):
            vertical_strings[j][position] = char
    
    # Convert lists to strings and trim trailing *
    vertical_strings = [''.join(lst).rstrip('*') for lst in vertical_strings]
    
    for s in vertical_strings:
        print(s)

if __name__ == "__main__":
    main()