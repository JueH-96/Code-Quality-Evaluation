def main():
    S = input().rstrip()
    # Since S is guaranteed to end with '2023', just replace the last character with '4'
    result = S[:-1] + '4'
    print(result)

if __name__ == "__main__":
    main()