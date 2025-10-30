def main():
    N = input().strip()
    # Check each adjacent pair of digits
    for i in range(len(N) - 1):
        if not (N[i] > N[i + 1]):
            print("No")
            return
    print("Yes")

if __name__ == "__main__":
    main()