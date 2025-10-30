# YOUR CODE HERE
S = input().strip()
n = len(S)
# Check if the length is even
if n % 2 != 0:
    print("No")
    exit()
# Check that each pair of characters are the same
for i in range(0, n, 2):
    if S[i] != S[i+1]:
        print("No")
        exit()
# Check that each character appears exactly twice
from collections import defaultdict
char_count = defaultdict(int)
for char in S:
    char_count[char] += 1
for char, count in char_count.items():
    if count != 2:
        print("No")
        exit()
print("Yes")