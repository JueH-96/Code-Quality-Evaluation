def main():
    import sys
    data = sys.stdin.read().strip().split()
    A, B, C = map(int, data)
    
    # If all three are equal, we can form three groups each containing one of the numbers
    if A == B == C:
        print("Yes")
        return
    
    S = A + B + C
    
    # Check if we can partition into 2 groups of sum S/2
    # That requires S to be even, and one of the subsets of {A,B,C} must have sum S/2
    if S % 2 == 0:
        half = S // 2
        # Check all subsets of size 1 or 2 that might sum to half
        if A == half or B == half or C == half \
           or (A + B) == half or (B + C) == half or (A + C) == half:
            print("Yes")
            return
    
    # If neither condition is satisfied, print No
    print("No")


# Do not forget to call main()!
main()