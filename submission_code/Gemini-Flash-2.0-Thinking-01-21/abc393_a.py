# Read the two strings from standard input
s1, s2 = input().split()

# Determine the bad oyster type based on the sickness status
# Takahashi ate oysters 1 and 2.
# Aoki ate oysters 1 and 3.
# Exactly one oyster type (1, 2, 3, or 4) causes sickness.

# If S_1 (Takahashi) is sick and S_2 (Aoki) is sick:
# Takahashi ate 1 and 2, got sick -> Bad oyster is 1 or 2.
# Aoki ate 1 and 3, got sick -> Bad oyster is 1 or 3.
# For both to be true, the bad oyster must be in {1, 2} intersect {1, 3} = {1}.
# If oyster 1 is bad: T gets sick (ate 1), A gets sick (ate 1). Matches s1='sick', s2='sick'.
if s1 == 'sick' and s2 == 'sick':
    print(1)
# If S_1 (Takahashi) is sick and S_2 (Aoki) is fine:
# Takahashi ate 1 and 2, got sick -> Bad oyster is 1 or 2.
# Aoki ate 1 and 3, was fine -> Bad oyster is NOT 1 and NOT 3. Possible bad oysters not eaten by Aoki are 2 or 4.
# For both conditions, the bad oyster must be in {1, 2} intersect {2, 4} = {2}.
# If oyster 2 is bad: T gets sick (ate 2), A gets fine (didn't eat 2). Matches s1='sick', s2='fine'.
elif s1 == 'sick' and s2 == 'fine':
    print(2)
# If S_1 (Takahashi) is fine and S_2 (Aoki) is sick:
# Takahashi ate 1 and 2, was fine -> Bad oyster is NOT 1 and NOT 2. Possible bad oysters not eaten by Takahashi are 3 or 4.
# Aoki ate 1 and 3, got sick -> Bad oyster is 1 or 3.
# For both conditions, the bad oyster must be in {3, 4} intersect {1, 3} = {3}.
# If oyster 3 is bad: T gets fine (didn't eat 3), A gets sick (ate 3). Matches s1='fine', s2='sick'.
elif s1 == 'fine' and s2 == 'sick':
    print(3)
# If S_1 (Takahashi) is fine and S_2 (Aoki) is fine:
# Takahashi ate 1 and 2, was fine -> Bad oyster is NOT 1 and NOT 2 (3 or 4).
# Aoki ate 1 and 3, was fine -> Bad oyster is NOT 1 and NOT 3 (2 or 4).
# For both conditions, the bad oyster must be in {3, 4} intersect {2, 4} = {4}.
# If oyster 4 is bad: T gets fine (didn't eat 4), A gets fine (didn't eat 4). Matches s1='fine', s2='fine'.
elif s1 == 'fine' and s2 == 'fine':
    print(4)