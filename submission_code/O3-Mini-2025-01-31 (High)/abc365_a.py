def main():
    import sys
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    Y = int(input_data[0])
    
    # Determine leap year based on rules:
    if Y % 400 == 0:
        print(366)
    elif Y % 100 == 0:
        print(365)
    elif Y % 4 == 0:
        print(366)
    else:
        print(365)

if __name__ == '__main__':
    main()