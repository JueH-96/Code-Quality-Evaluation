def main():
    A, B, C = map(int, input().split())
    total = A + B + C
    if total % 2 == 0 and max(A, B, C) == total // 2:
        print("Yes")
    elif total % 3 == 0 and A == total // 3 and B == total // 3 and C == total // 3:
        print("Yes")
    else:
        print("No")

if __name__ == '__main__':
    main()