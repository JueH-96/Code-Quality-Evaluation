def main():
    import sys
    import re
    input_data = sys.stdin.read().strip()
    S = input_data.split()[0] if input_data else ''
    # Check if S can be written as A* followed by B* followed by C*
    if re.fullmatch(r'A*B*C*', S):
        print("Yes")
    else:
        print("No")

if __name__ == '__main__':
    main()