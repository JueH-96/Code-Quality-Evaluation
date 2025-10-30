def main():
    A, B, C = map(int, input().split())
    total = A + B + C
    if A == B == C:
        print("Yes")
    elif total % 2 == 0:
        half = total // 2
        if A == half or B == half or C == half:
            print("Yes")
        else:
            print("No")
    else:
        print("No")

if __name__ == '__main__':
    main()