# YOUR CODE HERE
def find_difference_position(S, T):
    # Find the minimum length of the two strings
    min_length = min(len(S), len(T))
    
    # Compare characters up to the minimum length
    for i in range(min_length):
        if S[i] != T[i]:
            return i + 1  # Return the 1-based index of the first differing character
    
    # If all characters up to min_length are the same, check if lengths differ
    if len(S) != len(T):
        return min_length + 1  # Return the position where one string ends and the other continues
    
    # If both strings are equal
    return 0

if __name__ == "__main__":
    import sys
    input = sys.stdin.read
    data = input().strip().split()
    S = data[0]
    T = data[1]
    result = find_difference_position(S, T)
    print(result)