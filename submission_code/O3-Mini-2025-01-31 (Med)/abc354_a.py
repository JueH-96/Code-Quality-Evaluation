def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    if not input_data:
        return
    H = int(input_data[0])
    
    height = 0
    day = 0  # day 0: morning height is 0 cm.
    while height <= H:
        day += 1
        # Increase height by 2^(day-1) during the night of day-1.
        height += 2**(day-1)
        
    print(day)

if __name__ == '__main__':
    main()