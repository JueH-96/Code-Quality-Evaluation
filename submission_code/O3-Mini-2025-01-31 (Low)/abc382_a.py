def main():
    import sys
    input_data = sys.stdin.read().splitlines()
    if not input_data:
        return

    # Parse first line to get N and D
    first_line = input_data[0].split()
    N = int(first_line[0])
    D = int(first_line[1])
    
    # Parse the state string S
    S = input_data[1].strip()
    
    # Count the initially empty boxes ('.')
    initially_empty = S.count('.')
    
    # After D days, D cookies have been eaten.
    # Each eaten cookie makes its box empty.
    # Hence, total number of empty boxes = initially empty boxes + D.
    result = initially_empty + D
    
    print(result)

if __name__ == '__main__':
    main()