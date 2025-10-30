def main():
    A, B = map(int, input().split())
    rowA = (A - 1) // 3
    rowB = (B - 1) // 3
    if rowA == rowB and B - A == 1:
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    main()