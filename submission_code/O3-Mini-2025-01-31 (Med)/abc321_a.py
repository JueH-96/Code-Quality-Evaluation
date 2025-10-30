def main():
    import sys
    input_data = sys.stdin.read().strip()
    if not input_data:
        return
    # Read N as a string for ease of processing digits.
    s = input_data.strip()
    
    # A single digit is always valid as per the problem definition.
    for i in range(len(s) - 1):
        if s[i] <= s[i + 1]:
            print("No")
            return
    print("Yes")

if __name__ == "__main__":
    main()