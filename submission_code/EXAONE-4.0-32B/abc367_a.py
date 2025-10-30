def main():
    A, B, C = map(int, input().split())
    if B < C:
        if B < A < C:
            print("No")
        else:
            print("Yes")
    else:
        if A >= B or A < C:
            print("No")
        else:
            print("Yes")

if __name__ == "__main__":
    main()