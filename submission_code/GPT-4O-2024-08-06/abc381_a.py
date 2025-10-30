# YOUR CODE HERE
def is_11_22_string(N, S):
    # Check if the length of the string is odd
    if N % 2 == 0:
        return "No"
    
    # Calculate the middle index
    mid_index = (N + 1) // 2
    
    # Check the first part of the string (all '1's)
    if S[:mid_index - 1] != '1' * (mid_index - 1):
        return "No"
    
    # Check the middle character is '/'
    if S[mid_index - 1] != '/':
        return "No"
    
    # Check the last part of the string (all '2's)
    if S[mid_index:] != '2' * (N - mid_index):
        return "No"
    
    # If all conditions are satisfied
    return "Yes"

if __name__ == "__main__":
    import sys
    input = sys.stdin.read
    data = input().strip().split()
    N = int(data[0])
    S = data[1]
    result = is_11_22_string(N, S)
    print(result)