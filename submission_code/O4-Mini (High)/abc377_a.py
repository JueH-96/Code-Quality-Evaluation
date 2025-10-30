def main():
    S = input().strip()
    # Check if S can be rearranged to "ABC"
    if sorted(S) == sorted("ABC"):
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    main()