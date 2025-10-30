def main():
    import sys
    input = sys.stdin.read
    data = input().strip().splitlines()
    
    N = int(data[0])  # Read the number of strings
    strings = data[1:N+1]  # Read the strings
    
    # Sort the strings by their lengths
    sorted_strings = sorted(strings, key=len)
    
    # Concatenate the sorted strings
    result = ''.join(sorted_strings)
    
    # Print the result
    print(result)

if __name__ == "__main__":
    main()