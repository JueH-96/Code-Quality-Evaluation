import sys

def generate_dragon_string(N):
    # Construct the Dragon String based on the given level N
    dragon_string = 'L' + 'o' * N + 'n' + 'g'
    return dragon_string

def main():
    # Read input from stdin
    input = sys.stdin.read()

    # Parse the input to get the integer N
    N = int(input.strip())

    # Generate the Dragon String of level N
    result = generate_dragon_string(N)

    # Print the result to stdout
    print(result)

if __name__ == "__main__":
    main()