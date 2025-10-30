def main():
    S = input().strip()
    # Change the last character (which is '3') to '4'
    result = S[:-1] + '4'
    print(result)

if __name__ == "__main__":
    main()