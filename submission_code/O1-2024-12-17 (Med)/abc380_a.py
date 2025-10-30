def main():
    N = input().strip()

    # Check the count of digits '1', '2', and '3'
    count_1 = N.count('1')
    count_2 = N.count('2')
    count_3 = N.count('3')

    # Verify conditions
    if count_1 == 1 and count_2 == 2 and count_3 == 3:
        print("Yes")
    else:
        print("No")

# Do not remove or modify the following call
main()