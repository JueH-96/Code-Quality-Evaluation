# YOUR CODE HERE
def solve():
    N_str = input()

    # Count occurrences of '1', '2', '3'
    count1 = N_str.count('1')
    count2 = N_str.count('2')
    count3 = N_str.count('3')

    # Check if all conditions are met
    if count1 == 1 and count2 == 2 and count3 == 3:
        # The problem states N is a 6-digit number.
        # The sum of counts 1 + 2 + 3 = 6.
        # If these counts are met, all 6 digits are '1', '2', or '3'
        # in the specified frequencies. No other digits can be present.
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    solve()