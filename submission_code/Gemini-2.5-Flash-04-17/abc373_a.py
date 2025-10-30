import sys

count = 0
# The problem asks for the number of integers i (1 <= i <= 12)
# such that the length of the i-th string S_i is equal to i.
# We will iterate from i = 1 to 12.
for i in range(1, 13):
    # Read the i-th string. In the input, S_i is the i-th line.
    s_i = sys.stdin.readline().strip()
    
    # Check if the length of the string read (S_i) is equal to i.
    if len(s_i) == i:
        # If the condition is satisfied, increment the counter.
        count += 1

# After checking all 12 strings, print the total count.
print(count)