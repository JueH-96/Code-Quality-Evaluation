# YOUR CODE HERE
s1, s2 = input().split()

if s1 == "sick" and s2 == "sick":
    # Both sick - must be oyster 1 (common to both)
    print(1)
elif s1 == "sick" and s2 == "fine":
    # Takahashi sick, Aoki fine - must be oyster 2 (only Takahashi ate)
    print(2)
elif s1 == "fine" and s2 == "sick":
    # Takahashi fine, Aoki sick - must be oyster 3 (only Aoki ate)
    print(3)
else:  # s1 == "fine" and s2 == "fine"
    # Both fine - must be oyster 4 (neither ate)
    print(4)