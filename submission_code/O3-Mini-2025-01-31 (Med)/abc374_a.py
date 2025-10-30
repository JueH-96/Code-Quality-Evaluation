def main():
    # Read input string and remove any whitespace like newlines
    s = input().strip()
    # Check if the string ends with "san"
    if s.endswith("san"):
        print("Yes")
    else:
        print("No")

# Call the main function to execute the program
if __name__ == "__main__":
    main()