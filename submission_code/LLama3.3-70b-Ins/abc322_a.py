def find_abc_position():
    # Read the length of the string
    N = int(input())

    # Read the string
    S = input()

    # Find the position of the first occurrence of 'ABC'
    for i in range(N - 2):
        if S[i:i+3] == 'ABC':
            return i + 1

    # If 'ABC' is not found, return -1
    return -1

# Print the result
print(find_abc_position())