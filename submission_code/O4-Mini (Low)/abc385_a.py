def main():
    import sys
    A, B, C = map(int, sys.stdin.read().split())
    S = A + B + C

    # Check k = 2 groups
    if S % 2 == 0:
        half = S // 2
        # Any single element equals half?
        if A == half or B == half or C == half:
            print("Yes")
            return
        # Any pair sums to half?
        if A + B == half or A + C == half or B + C == half:
            print("Yes")
            return

    # Check k = 3 groups
    # Only possible if all three are the same and S divisible by 3
    if S % 3 == 0:
        third = S // 3
        if A == third and B == third and C == third:
            print("Yes")
            return

    # Otherwise no valid partition
    print("No")

if __name__ == "__main__":
    main()