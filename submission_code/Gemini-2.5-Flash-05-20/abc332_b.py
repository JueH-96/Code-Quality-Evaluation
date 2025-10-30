import sys

def solve():
    K, G_capacity, M_capacity = map(int, sys.stdin.readline().split())

    g_amount = 0  # Current amount of water in the glass
    m_amount = 0  # Current amount of water in the mug

    for _ in range(K):
        # Rule 1: When the glass is filled with water, discard all the water from the glass.
        if g_amount == G_capacity:
            g_amount = 0
        # Rule 2: Otherwise, if the mug is empty, fill the mug with water.
        elif m_amount == 0:
            m_amount = M_capacity
        # Rule 3: Otherwise, transfer water from the mug to the glass until the mug is empty or the glass is filled with water.
        else:
            # Calculate the space available in the glass
            space_in_glass = G_capacity - g_amount
            
            # Determine the amount of water to transfer
            # It's the minimum of:
            # 1. The water currently in the mug (m_amount)
            # 2. The remaining space in the glass (space_in_glass)
            transfer_amount = min(m_amount, space_in_glass)

            g_amount += transfer_amount
            m_amount -= transfer_amount
    
    # Print the final amounts in the glass and mug, separated by a space.
    print(g_amount, m_amount)

# Call the solve function to execute the program logic
solve()