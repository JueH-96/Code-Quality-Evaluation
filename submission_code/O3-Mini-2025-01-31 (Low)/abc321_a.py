def main():
    import sys
    input_str = sys.stdin.read().strip()
    if not input_str:
        return
    # Remove any extra whitespace/newlines
    tokens = input_str.split()
    N_str = tokens[0]
    
    # Checking the digits if in strictly descending order
    is_321_like = True
    for i in range(len(N_str) - 1):
        if not (N_str[i] > N_str[i+1]):
            is_321_like = False
            break
    
    if is_321_like:
        print("Yes")
    else:
        print("No")
    
if __name__ == '__main__':
    main()