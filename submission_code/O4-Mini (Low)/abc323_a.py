def main():
    S = input().strip()
    # Check even positions in 1-based indexing -> odd indices in 0-based
    for i in range(1, 16, 2):
        if S[i] != '0':
            print("No")
            return
    print("Yes")

if __name__ == "__main__":
    main()