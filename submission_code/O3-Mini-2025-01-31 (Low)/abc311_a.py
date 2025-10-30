def main():
    import sys
    input_data = sys.stdin.read().split()
    # First token is N, second token is S
    N = int(input_data[0])
    S = input_data[1]
    
    # Set to track appearance of letters A, B, C
    seen = set()
    # Check for each character
    for i, char in enumerate(S, start=1):
        seen.add(char)
        if len(seen) == 3:
            print(i)
            return

if __name__ == '__main__':
    main()