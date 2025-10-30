def main():
    S = input().strip()
    # We need to check the characters at 1-based even positions i = 2,4,6,...,16
    # In Python (0-based), those correspond to indices 1,3,5,...,15.
    for i in range(1, 16, 2):
        if S[i] != '0':
            print("No")
            return
    print("Yes")

# Don't forget to call main()
main()