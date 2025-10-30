def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    A, B, C = map(int, data)
    total = A + B + C

    # Case 1: Partition into 2 groups
    # For 2 groups to have equal sum, one group must sum to total/2.
    # With only three numbers, one group will be a single element and the other will be the remaining two.
    # Hence, we can check if any number equals total/2.
    if total % 2 == 0:
        half = total // 2
        if A == half or B == half or C == half:
            sys.stdout.write("Yes")
            return
            
    # Case 2: Partition into 3 groups (each element separately)
    # In this case, each number must be equal (i.e. A == B == C).
    if A == B == C:
        sys.stdout.write("Yes")
        return

    sys.stdout.write("No")
    
if __name__ == '__main__':
    main()