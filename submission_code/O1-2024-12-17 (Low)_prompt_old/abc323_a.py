def solve():
    S = input().strip()
    # We need to check the characters at even positions (2, 4, ..., 16).
    # In Python's 0-based indexing, these correspond to indices 1, 3, 5, ..., 15.
    for i in range(1, 16, 2):
        if S[i] != '0':
            print("No")
            return
    print("Yes")

def main():
    solve()

if __name__ == "__main__":
    main()