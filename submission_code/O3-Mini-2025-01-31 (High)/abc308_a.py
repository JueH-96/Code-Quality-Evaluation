def main():
    import sys
    # Read all the input from standard input and split into tokens.
    tokens = sys.stdin.read().split()
    # Convert tokens into integers (expecting 8 integers).
    S = list(map(int, tokens))
    
    # Check condition 1: Monotonically non-decreasing sequence.
    condition1 = all(S[i] <= S[i+1] for i in range(7))
    
    # Check condition 2: All values are between 100 and 675, inclusive.
    condition2 = all(100 <= num <= 675 for num in S)
    
    # Check condition 3: All values are multiples of 25.
    condition3 = all(num % 25 == 0 for num in S)
    
    # If all conditions hold, print "Yes"; otherwise, print "No".
    if condition1 and condition2 and condition3:
        print("Yes")
    else:
        print("No")
        
if __name__ == '__main__':
    main()