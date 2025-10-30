def main():
    A, B = map(int, input().split())
    if (A - 1) // 3 == (B - 1) // 3 and B == A + 1:
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    main()