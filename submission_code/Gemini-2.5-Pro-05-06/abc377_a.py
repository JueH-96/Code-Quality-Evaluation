# Read the input string S
S = input()

# Sort the characters of S and join them back into a string
sorted_S = "".join(sorted(S))

# Check if the sorted string is "ABC"
if sorted_S == "ABC":
    print("Yes")
else:
    print("No")