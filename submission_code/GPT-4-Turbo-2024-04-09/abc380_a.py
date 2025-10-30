# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    N = input().strip()
    
    # Count the occurrences of 1, 2, and 3
    count_1 = N.count('1')
    count_2 = N.count('2')
    count_3 = N.count('3')
    
    # Check the conditions
    if count_1 == 1 and count_2 == 2 and count_3 == 3:
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    main()