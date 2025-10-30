def main():
    # Read the eight integers from standard input
    S = list(map(int, input().split()))
    
    # Condition 1: The sequence is monotonically non-decreasing.
    cond1 = all(S[i] <= S[i+1] for i in range(7))
    
    # Condition 2: All numbers are between 100 and 675, inclusive.
    cond2 = all(100 <= x <= 675 for x in S)
    
    # Condition 3: All numbers are multiples of 25.
    cond3 = all(x % 25 == 0 for x in S)
    
    # Print "Yes" if all conditions are met, otherwise "No"
    if cond1 and cond2 and cond3:
        print("Yes")
    else:
        print("No")

# Do not forget to call main function
if __name__ == "__main__":
    main()