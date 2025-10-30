# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()
N = int(data[0])
M = int(data[1])
S = data[2]

# Initialize variables
plain_tshirts = M
logo_tshirts = 0
worn_tshirts = []

# Iterate through the schedule
for day in S:
    if day == '0':
        # No plans, wash all worn T-shirts
        worn_tshirts = []
    elif day == '1':
        # Plan to go out for a meal
        if plain_tshirts > 0:
            plain_tshirts -= 1
            worn_tshirts.append('plain')
        elif logo_tshirts > 0:
            logo_tshirts -= 1
            worn_tshirts.append('logo')
        else:
            logo_tshirts += 1
            worn_tshirts.append('logo')
    elif day == '2':
        # Plan to attend a competitive programming event
        if logo_tshirts > 0:
            logo_tshirts -= 1
            worn_tshirts.append('logo')
        else:
            logo_tshirts += 1
            worn_tshirts.append('logo')

# Output the minimum number of T-shirts to buy
print(logo_tshirts)