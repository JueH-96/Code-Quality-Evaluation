def main():
    A, B = map(int, input().split())
    # Check if they are horizontally adjacent:
    # That happens when B == A + 1 AND A is not the last in its row (A % 3 != 0)
    if B == A + 1 and A % 3 != 0:
        print("Yes")
    else:
        print("No")

main()