def main():
    import sys
    # Read the input and parse the three integers A, B, C.
    data = sys.stdin.read().strip().split()
    if not data:
        return
    A, B, C = map(int, data)
    
    # Case 1: Check if all three numbers are equal.
    # In this case, we can partition into three groups: {A}, {B}, and {C}.
    if A == B == C:
        print("Yes")
        return
    
    # Case 2: Partition into 2 groups.
    # There are only 3 possible ways (one number vs the other two):
    #  - Group1: A, Group2: B and C   -> the sum equality condition: A == B + C
    #  - Group1: B, Group2: A and C   -> the condition: B == A + C
    #  - Group1: C, Group2: A and B   -> the condition: C == A + B
    if A == B + C or B == A + C or C == A + B:
        print("Yes")
    else:
        print("No")

if __name__ == '__main__':
    main()