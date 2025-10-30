# Read input values
N, M = map(int, input().split())
H = list(map(int, input().split()))

# Initialize the number of aliens who can disinfect all their hands
aliens_with_clean_hands = 0

# Loop through each alien and attempt to disinfect their hands
for h in H:
    if M >= h:
        M -= h
        aliens_with_clean_hands += 1
    else:
        break

# Output the result
print(aliens_with_clean_hands)