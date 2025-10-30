def main():
    N = input().strip()
    # Count occurrences of '1', '2', and '3'
    if N.count('1') == 1 and N.count('2') == 2 and N.count('3') == 3:
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    main()