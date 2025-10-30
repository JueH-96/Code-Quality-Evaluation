# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()
S = data[0]

# Split the string by '.s' and get the last substring
last_substring = S.split('.s')[-1]

# Print the last substring
print(last_substring)