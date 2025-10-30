# Read the input string S
S = input()

# Convert the string S to a list of characters, sort the list, and then join the characters back into a string.
# If S contains 'A', 'B', and 'C' (and nothing else, as its length is 3),
# then sorting its characters will always result in the string "ABC".
sorted_S = "".join(sorted(list(S)))

# Check if the sorted string is "ABC"
if sorted_S == "ABC":
    print("Yes")
else:
    print("No")