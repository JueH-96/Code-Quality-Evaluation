# Read the input string from standard input
S = input().strip()

# Check if the string is in the range of valid contest abbreviations
# and not equal to the excluded "ABC316"
if S[:3] == "ABC" and 1 <= int(S[3:]) <= 349 and S != "ABC316":
    print("Yes")
else:
    print("No")