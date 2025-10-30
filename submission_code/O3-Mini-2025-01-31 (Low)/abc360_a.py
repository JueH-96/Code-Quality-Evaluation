def main():
    import sys
    input_str = sys.stdin.read().strip()
    # Ensure we deal with just the first non-empty line
    if not input_str:
        return
    # Use first line as input string S
    S = input_str.split()[0]
    
    # Find the positions - Python index from 0 is sufficient.
    pos_R = S.find('R')
    pos_M = S.find('M')
    
    # Check if plate of rice (R) is to the left of miso soup (M)
    if pos_R < pos_M:
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    main()