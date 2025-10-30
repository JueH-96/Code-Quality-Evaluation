import sys

# Read the input line from stdin
line = sys.stdin.readline().strip()

# Split the line into the two strings S_1 and S_2
s1, s2 = line.split()

# Determine the problematic oyster based on the sickness status
# There are 4 types of oysters: 1, 2, 3, 4. Exactly one type causes trouble.
# Takahashi ate oysters 1 and 2.
# Aoki ate oysters 1 and 3.

# If Takahashi got sick (S_1 == "sick"), the troublesome oyster is one he ate: 1 or 2.
# If Takahashi did not get sick (S_1 == "fine"), the troublesome oyster is one he did NOT eat: 3 or 4.

# If Aoki got sick (S_2 == "sick"), the troublesome oyster is one he ate: 1 or 3.
# If Aoki did not get sick (S_2 == "fine"), the troublesome oyster is one he did NOT eat: 2 or 4.

# We combine the information from S_1 and S_2 to find the unique troublesome oyster.

# Case 1: Takahashi sick, Aoki sick
# Troublesome is in {1, 2} AND in {1, 3}. The intersection is {1}.
if s1 == "sick" and s2 == "sick":
    print(1)
# Case 2: Takahashi sick, Aoki fine
# Troublesome is in {1, 2} AND NOT in {1, 3} (meaning in {2, 4}). The intersection is {2}.
elif s1 == "sick" and s2 == "fine":
    print(2)
# Case 3: Takahashi fine, Aoki sick
# Troublesome is NOT in {1, 2} (meaning in {3, 4}) AND in {1, 3}. The intersection is {3}.
elif s1 == "fine" and s2 == "sick":
    print(3)
# Case 4: Takahashi fine, Aoki fine
# Troublesome is NOT in {1, 2} (meaning in {3, 4}) AND NOT in {1, 3} (meaning in {2, 4}). The intersection is {4}.
elif s1 == "fine" and s2 == "fine":
    print(4)