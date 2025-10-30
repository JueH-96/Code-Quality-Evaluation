def main():
    S = input().strip()
    # Check if the sorted characters of S match "ABC"
    if sorted(S) == list("ABC"):
        print("Yes")
    else:
        print("No")

# Call the main function
main()