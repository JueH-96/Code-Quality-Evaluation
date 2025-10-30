def main():
    import sys

    # Read eight integers from standard input.
    S = list(map(int, sys.stdin.read().split()))
    
    # Condition 1: Non-decreasing sequence
    non_decreasing = all(S[i] <= S[i+1] for i in range(7))
    
    # Condition 2: Each between 100 and 675 inclusive
    in_range = all(100 <= x <= 675 for x in S)
    
    # Condition 3: Each is a multiple of 25
    multiples_of_25 = all(x % 25 == 0 for x in S)
    
    # Print the result
    if non_decreasing and in_range and multiples_of_25:
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    main()