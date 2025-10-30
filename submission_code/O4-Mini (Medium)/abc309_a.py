def main():
    A, B = map(int, input().split())
    # Compute zero-based positions
    posA = A - 1
    posB = B - 1
    rowA, colA = divmod(posA, 3)
    rowB, colB = divmod(posB, 3)
    # Check horizontal adjacency: same row and columns differ by 1
    if rowA == rowB and abs(colA - colB) == 1:
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    main()