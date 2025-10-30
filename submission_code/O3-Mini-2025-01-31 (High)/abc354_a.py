def main():
    import sys
    # Read the single integer input
    input_text = sys.stdin.read().strip().split()
    H = int(input_text[0])
    
    # Initialize plant height at 0 cm (day 0 morning)
    height = 0
    day = 0
    
    # Each night, the plant grows by 2^(current day) cm.
    # In the morning, we check if the plant's height is strictly greater than H.
    while height <= H:
        height += 2 ** day   # Growth happens on the night of the current day
        day += 1             # Next morning, we check the updated height
        
    print(day)

# Call main function to execute the program.
if __name__ == '__main__':
    main()