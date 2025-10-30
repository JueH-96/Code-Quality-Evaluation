# YOUR CODE HERE
N, M = map(int, input().split())
S = input()

# Initialize the number of T-shirts needed to 0
t_shirts_needed = 0

# Initialize the number of logo T-shirts available to M
logo_t_shirts_available = M

# Iterate over the days
for day in S:
    # If the day is a meal day
    if day == '1':
        # If there are no logo T-shirts available
        if logo_t_shirts_available == 0:
            # Increase the number of T-shirts needed
            t_shirts_needed += 1
        else:
            # Decrease the number of logo T-shirts available
            logo_t_shirts_available -= 1
    # If the day is a competitive programming event day
    elif day == '2':
        # If there are no logo T-shirts available
        if logo_t_shirts_available == 0:
            # Increase the number of T-shirts needed
            t_shirts_needed += 1
        else:
            # Decrease the number of logo T-shirts available
            logo_t_shirts_available -= 1
    # If the day is a no plan day
    else:
        # Reset the number of logo T-shirts available to M
        logo_t_shirts_available = M

# Print the number of T-shirts needed
print(t_shirts_needed)