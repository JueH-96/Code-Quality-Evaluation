# Read the input strings S and T
S, T = input().split()

# Initialize the answer to No
answer = "No"

# Loop through all possible values of w
for w in range(1, len(S)):
    # Initialize a temporary string to store the concatenation of c-th characters
    temp = ""
    # Loop through the string S with step w to get the substrings
    for i in range(0, len(S), w):
        # If the length of the substring is at least c, concatenate the c-th character
        if i + w <= len(S):
            temp += S[i + w - 1]
    # Check if the temporary string equals T
    if temp == T:
        answer = "Yes"
        break

# Print the answer
print(answer)