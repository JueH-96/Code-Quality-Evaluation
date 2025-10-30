def main():
    A, B, C = map(int, input().split())
    total = A + B + C

    # Try partition into 2 groups: each must sum to total/2
    if total % 2 == 0:
        half = total // 2
        # If any single element equals half, the rest sum to half too
        if A == half or B == half or C == half:
            print("Yes")
            return

    # Try partition into 3 groups: each must sum to total/3.
    # With 3 items and 3 non-empty groups, each group is a single item.
    if total % 3 == 0:
        third = total // 3
        if A == third and B == third and C == third:
            print("Yes")
            return

    print("No")


if __name__ == "__main__":
    main()