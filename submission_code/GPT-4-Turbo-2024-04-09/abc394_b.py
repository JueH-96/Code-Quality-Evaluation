# YOUR CODE HERE
import sys
input = sys.stdin.read

def main():
    data = input().split()
    N = int(data[0])
    strings = data[1:]
    
    # Sort strings by their length
    sorted_strings = sorted(strings, key=len)
    
    # Concatenate sorted strings
    result = ''.join(sorted_strings)
    
    # Print the result
    print(result)

if __name__ == "__main__":
    main()