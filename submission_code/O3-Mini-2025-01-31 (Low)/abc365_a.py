def main():
    import sys
    input_data = sys.stdin.read().strip()
    if not input_data:
        return
    Y = int(input_data)
    
    # Check if Y is a leap year based on the given rules.
    if Y % 400 == 0:
        days = 366
    elif Y % 100 == 0:
        days = 365
    elif Y % 4 == 0:
        days = 366
    else:
        days = 365
        
    sys.stdout.write(str(days))
    
if __name__ == '__main__':
    main()