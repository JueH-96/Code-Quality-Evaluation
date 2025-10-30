def main():
    N = input().strip()  # Read the input number as a string.
    # Check the required conditions: 
    # digit '1' occurs exactly once, '2' exactly twice and '3' exactly three times.
    if N.count('1') == 1 and N.count('2') == 2 and N.count('3') == 3:
        print("Yes")
    else:
        print("No")

if __name__ == '__main__':
    main()