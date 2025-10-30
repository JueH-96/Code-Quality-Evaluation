def main():
    import sys
    input_data = sys.stdin.read().strip()
    if not input_data:
        return
    
    S = input_data.strip()
    # Convert the last three characters to an integer.
    n = int(S[3:])
    
    # Check if n is between 1 and 349 inclusive, except n is 316.
    if 1 <= n <= 349 and n != 316:
        print("Yes")
    else:
        print("No")

if __name__ == '__main__':
    main()