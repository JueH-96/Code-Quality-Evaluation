def main():
    import sys
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    S = data[1]

    for i in range(N - 1):
        if (S[i] == 'a' and S[i+1] == 'b') or (S[i] == 'b' and S[i+1] == 'a'):
            print("Yes")
            return

    print("No")

# Call the main function
if __name__ == "__main__":
    main()