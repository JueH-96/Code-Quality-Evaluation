def main():
    S = input().strip()
    # Check even positions (1-based): indices 2,4,...,16 => 0-based indices 1,3,...,15
    for i in range(1, 16, 2):
        if S[i] != '0':
            print("No")
            return
    print("Yes")

if __name__ == "__main__":
    main()