# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().strip().split('
')
    
    N = int(data[0])
    strings = data[1:N+1]
    
    # Sort strings by their length
    strings.sort(key=len)
    
    # Concatenate the sorted strings
    result = ''.join(strings)
    
    # Print the result
    print(result)

if __name__ == "__main__":
    main()