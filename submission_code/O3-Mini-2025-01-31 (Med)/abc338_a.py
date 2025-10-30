def main():
    import sys
    input_data = sys.stdin.read().strip()
    if not input_data:
        return
    S = input_data.split()[0]
    # Check if the first character is uppercase and the rest are lowercase
    if S[0].isupper() and S[1:].islower():
        print("Yes")
    else:
        print("No")

if __name__ == '__main__':
    main()