def main():
    S = input().strip()
    if sorted(S) == sorted("ABC"):
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    main()