def main():
    S = input().strip()
    # Check if by rearranging S we can get "ABC"
    if sorted(S) == ["A", "B", "C"]:
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    main()