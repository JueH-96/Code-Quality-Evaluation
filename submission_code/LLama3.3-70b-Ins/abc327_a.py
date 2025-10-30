def check_adjacent_ab():
    # Read the length of the string
    N = int(input())

    # Read the string
    S = input()

    # Check for adjacent occurrences of 'a' and 'b'
    for i in range(N - 1):
        if (S[i] == 'a' and S[i + 1] == 'b') or (S[i] == 'b' and S[i + 1] == 'a'):
            print("Yes")
            return

    # If no adjacent occurrences are found, print "No"
    print("No")

# Call the function
check_adjacent_ab()