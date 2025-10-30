# YOUR CODE HERE
def check_string_format(S):
    # Check if the first character is uppercase
    if S[0].isupper():
        # Check if all other characters are lowercase
        if S[1:].islower():
            return "Yes"
    return "No"

if __name__ == "__main__":
    import sys
    input = sys.stdin.read
    S = input().strip()
    result = check_string_format(S)
    print(result)