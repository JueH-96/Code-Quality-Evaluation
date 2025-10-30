# Read input
N = int(input())
S = input()

# Initialize counters for A, B, and C
a_count = 0
b_count = 0
c_count = 0

# Iterate through the string and count the characters
for i in range(N):
    if S[i] == 'A':
        a_count += 1
    elif S[i] == 'B':
        b_count += 1
    elif S[i] == 'C':
        c_count += 1
    
    # Check if all A, B, and C have appeared at least once
    if a_count > 0 and b_count > 0 and c_count > 0:
        print(i + 1)
        break