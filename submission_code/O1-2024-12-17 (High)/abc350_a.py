def main():
    S = input().strip()  # Read the input string
    number_part = int(S[3:])  # Extract the numeric part

    # Check if the number part is between 1 and 349 (inclusive) but not 316
    if 1 <= number_part <= 349 and number_part != 316:
        print("Yes")
    else:
        print("No")

# Do not forget to call main() 
main()