import sys

def solve():
    """
    Reads input, solves the AtCoder T-shirt problem, and prints the output.
    """
    # Read problem parameters N (days) and M (plain T-shirts).
    try:
        n, m = map(int, sys.stdin.readline().split())
        # Read the schedule string S.
        s = sys.stdin.readline().strip()
    except (IOError, ValueError):
        # This handles potential empty input on some platforms.
        return

    # plain_used: Tracks plain shirts used in the current wash cycle.
    plain_used = 0
    # logo_used: Tracks logo shirts used in the current wash cycle.
    logo_used = 0
    
    # ans: Stores the maximum number of logo shirts needed at any point. This is the answer.
    ans = 0

    # Iterate through the schedule day by day.
    for event in s:
        if event == '0':
            # Wash day resets the cycle. All shirts are clean again.
            plain_used = 0
            logo_used = 0
        elif event == '1':  # Meal day
            # Prioritize using a plain shirt if one is available for this cycle.
            if plain_used < m:
                plain_used += 1
            else:
                # All plain shirts for this cycle are dirty; use a logo shirt.
                logo_used += 1
        elif event == '2':  # Event day
            # A logo shirt is mandatory.
            logo_used += 1
        
        # The number of logo shirts to own must be enough to cover the peak
        # usage in any cycle. We update our running maximum.
        ans = max(ans, logo_used)

    # Print the final calculated minimum number of logo shirts to buy.
    print(ans)

# Run the solution.
solve()