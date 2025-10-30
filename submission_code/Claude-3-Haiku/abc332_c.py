import sys

N, M = map(int, input().split())
S = input()

# Initialize the number of logo T-shirts needed
logo_tshirts = 0

# Initialize the number of plain T-shirts available
plain_tshirts = M

# Iterate through the schedule
for day in range(N):
    if S[day] == '1':  # Going out for a meal
        if plain_tshirts > 0:
            plain_tshirts -= 1
        else:
            logo_tshirts += 1
    elif S[day] == '2':  # Attending a competitive programming event
        logo_tshirts += 1
    else:  # No plans, wash all worn T-shirts
        plain_tshirts = M
        logo_tshirts = 0

print(logo_tshirts)