def main():
    import sys
    input = sys.stdin.readline
    # Read the number of strings
    n = int(input().strip())
    # Read each string and store in a list
    strings = [input().strip() for _ in range(n)]
    # Sort the strings by their length in ascending order
    strings.sort(key=len)
    # Concatenate the sorted strings
    result = ''.join(strings)
    # Print the result
    print(result)

if __name__ == '__main__':
    main()