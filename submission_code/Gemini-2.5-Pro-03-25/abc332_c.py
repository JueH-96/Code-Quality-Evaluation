import sys

# Define the main logic within a function
def solve():
    # Read N (number of days) and M (number of plain T-shirts) from the first line of input
    N, M = map(int, sys.stdin.readline().split())
    
    # Read the schedule string S from the second line of input
    S = sys.stdin.readline().strip()

    # Initialize state variables
    # used_plain: counts the number of plain T-shirts that have been used since the last wash day 
    # and are currently dirty.
    used_plain = 0
    
    # used_logo: counts the number of logo T-shirts that have been used since the last wash day
    # and are currently dirty. This variable tracks the current demand for logo T-shirts within a wash cycle.
    used_logo = 0
    
    # max_needed_logo: tracks the maximum value that used_logo reaches throughout the N days. 
    # This represents the peak requirement for logo T-shirts at any point in time.
    # Initialize to 0, as it's possible no logo shirts are needed.
    max_needed_logo = 0

    # Iterate through each day's schedule from day 1 to N
    for i in range(N):
        # Get the event type for the current day i (0-indexed)
        event = S[i]
        
        if event == '0':
            # Day with no plans: Wash all T-shirts.
            # All worn T-shirts become clean and available again.
            # Reset the counts of used (dirty) shirts for the next cycle.
            used_plain = 0
            used_logo = 0
        elif event == '1':
            # Meal day: Needs either a plain or a logo T-shirt.
            # Takahashi should prioritize using a plain T-shirt to save logo T-shirts for mandatory use.
            # Check if there is any clean plain T-shirt available.
            # The number of clean plain T-shirts is M - used_plain.
            # If M - used_plain > 0, or equivalently used_plain < M, a clean plain T-shirt is available.
            if used_plain < M:
                # Use a plain T-shirt. Increment the count of used (dirty) plain shirts.
                used_plain += 1
            else:
                # No clean plain T-shirts available. Must use a logo T-shirt.
                # Increment the count of used (dirty) logo shirts.
                used_logo += 1
        elif event == '2':
            # Competitive programming event day: Must use a logo T-shirt.
            # Increment the count of used (dirty) logo shirts.
            used_logo += 1
        
        # After processing the day's event and potentially incrementing used_logo,
        # update the maximum number of logo T-shirts needed concurrently so far.
        # This ensures we track the peak demand across the entire schedule.
        max_needed_logo = max(max_needed_logo, used_logo)

    # The final result is max_needed_logo. This is the minimum total number of logo T-shirts
    # Takahashi must possess to satisfy all requirements. Since he starts with 0 logo T-shirts,
    # this value is exactly the minimum number he needs to buy.
    print(max_needed_logo)

# Call the solve function to execute the logic when the script is run
solve()