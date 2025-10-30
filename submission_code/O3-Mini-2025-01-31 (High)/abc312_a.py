def main():
    # Read the input string and strip any extra whitespace
    S = input().strip()
    
    # Define the set of allowed strings
    allowed = {"ACE", "BDF", "CEG", "DFA", "EGB", "FAC", "GBD"}
    
    # Check if S is one of the allowed strings and print the result accordingly
    if S in allowed:
        print("Yes")
    else:
        print("No")

# Call the main function
if __name__ == "__main__":
    main()