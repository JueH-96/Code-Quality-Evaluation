def main():
    A, B = map(int, input().split())
    # Check they are consecutive and in the same row
    if B == A + 1 and (A - 1) // 3 == (B - 1) // 3:
        print("Yes")
    else:
        print("No")

# Call main function
main()