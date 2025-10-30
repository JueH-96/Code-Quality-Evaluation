# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    # Read the input
    data = input().strip()
    # Split the input into surname and first name
    S, T = data.split()
    # Print the required output
    print(f"{S} san")

if __name__ == "__main__":
    main()