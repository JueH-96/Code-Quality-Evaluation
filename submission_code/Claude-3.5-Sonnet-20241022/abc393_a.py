# Read input
s1, s2 = input().split()

# Takahashi ate 1,2
# Aoki ate 1,3
# One of 1,2,3,4 causes trouble

# If Takahashi is sick and Aoki is fine:
# - It can't be 1 (both would be sick)
# - Must be 2 (as Takahashi is sick but Aoki isn't)
if s1 == "sick" and s2 == "fine":
    print(2)

# If Takahashi is fine and Aoki is sick:
# - It can't be 1 (both would be sick)
# - Must be 3 (as Aoki is sick but Takahashi isn't)
elif s1 == "fine" and s2 == "sick":
    print(3)

# If both are sick:
# - Must be 1 (only oyster they both ate)
elif s1 == "sick" and s2 == "sick":
    print(1)

# If both are fine:
# - Can't be 1 (both ate it)
# - Can't be 2 (Takahashi ate it)
# - Can't be 3 (Aoki ate it)
# - Must be 4 (only remaining possibility)
else:  # s1 == "fine" and s2 == "fine"
    print(4)