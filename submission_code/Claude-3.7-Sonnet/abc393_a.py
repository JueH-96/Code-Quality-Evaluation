# Read input
s1, s2 = input().split()

# Takahashi ate oysters 1 and 2
# Aoki ate oysters 1 and 3

# Case analysis to determine which oyster causes trouble
if s1 == "sick" and s2 == "sick":
    # Both got sick, so it must be oyster 1 (the only one both ate)
    print(1)
elif s1 == "sick" and s2 == "fine":
    # Only Takahashi got sick, so it must be oyster 2 (only Takahashi ate it)
    print(2)
elif s1 == "fine" and s2 == "sick":
    # Only Aoki got sick, so it must be oyster 3 (only Aoki ate it)
    print(3)
else:  # s1 == "fine" and s2 == "fine"
    # Neither got sick, so it must be oyster 4 (neither ate it)
    print(4)