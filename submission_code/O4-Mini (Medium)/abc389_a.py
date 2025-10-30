def main():
    S = input().strip()
    # The string S is in the form "AxB" where A and B are digits
    a = int(S[0])
    b = int(S[2])
    print(a * b)

if __name__ == "__main__":
    main()