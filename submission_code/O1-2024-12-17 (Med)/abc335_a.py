def main():
    S = input().strip()
    # Since S is guaranteed to end with '2023', replace the last character '3' with '4'
    result = S[:-1] + '4'
    print(result)

if __name__ == "__main__":
    main()