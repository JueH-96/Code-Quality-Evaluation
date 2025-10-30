def main():
    # Read K, G, M from standard input
    K, G, M = map(int, input().split())

    # Initialize current water amounts
    current_glass_water = 0
    current_mug_water = 0

    # Perform the operation K times
    for _ in range(K):
        if current_glass_water == G:
            # Condition 1: Glass is full, discard water
            current_glass_water = 0
        elif current_mug_water == 0:
            # Condition 2: Mug is empty, fill mug
            current_mug_water = M
        else:
            # Condition 3: Transfer water from mug to glass
            
            # Calculate how much space is available in the glass
            space_in_glass = G - current_glass_water
            
            # Determine how much water to transfer:
            # It's the minimum of what's needed to fill the glass
            # and what's available in the mug.
            transfer_amount = min(space_in_glass, current_mug_water)
            
            current_glass_water += transfer_amount
            current_mug_water -= transfer_amount
            
    # Print the final amounts of water
    print(current_glass_water, current_mug_water)

if __name__ == '__main__':
    main()