def main():
    N = int(input().strip())
    S = input().strip()

    for i in range(N - 2):
        if S[i:i+3] == "ABC":
            print(i + 1)  # +1 for 1-based index
            return

    print(-1)  # If not found

# Call main function
main()