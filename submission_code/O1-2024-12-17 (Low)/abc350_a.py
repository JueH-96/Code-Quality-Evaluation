def main():
    S = input().strip()
    # Extract the numeric part, S is guaranteed to be of the form ABCddd where ddd are digits.
    num = int(S[3:])
    
    # Determine if num is in [1..315] or [317..349]
    # Note that 316 is excluded.
    if (1 <= num <= 315) or (317 <= num <= 349):
        print("Yes")
    else:
        print("No")

# Call the main function
main()