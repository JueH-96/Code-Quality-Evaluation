def main():
    import sys
    H = int(sys.stdin.readline().strip())
    
    day = 0
    height = 0
    growth = 1  # This represents 2^(day-1) for the current day
    
    while True:
        day += 1          # Move to the next day
        height += growth  # Increase height by this day's nightly growth
        if height > H:
            print(day)
            return
        growth <<= 1      # Equivalent to multiplying by 2

# Call main function
main()