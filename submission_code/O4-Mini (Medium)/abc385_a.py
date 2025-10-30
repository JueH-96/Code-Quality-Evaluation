def main():
    A, B, C = map(int, input().split())
    # Case 1: split into 2 groups: one element vs. the other two
    if A + B == C or A + C == B or B + C == A:
        print("Yes")
    # Case 2: split into 3 groups of one element each
    elif A == B == C:
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    main()