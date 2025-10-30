def count_substrings(s):
    count = 0
    for i in range(len(s)):
        for j in range(i+1, len(s)+1):
            count += 1
    return count

# Read input from stdin
s = input()

# Print the answer to stdout
print(count_substrings(s))