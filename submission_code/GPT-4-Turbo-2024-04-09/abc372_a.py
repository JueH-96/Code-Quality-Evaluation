# Reading input from standard input
import sys
input = sys.stdin.read

def main():
    S = input().strip()  # Read the input and strip any extraneous whitespace
    result = S.replace('.', '')  # Replace all occurrences of '.' with an empty string
    print(result)  # Print the result

if __name__ == "__main__":
    main()