def main():
    S = input().strip()
    # Check each even index from 2 to 16 (1-based)
    # In Python, string indices are 0-based, so we check S[i-1] for i in that range
    for i in range(2, 17, 2):
        if S[i-1] != '0':
            print("No")
            return
    print("Yes")

# Don't forget to call main()
main()