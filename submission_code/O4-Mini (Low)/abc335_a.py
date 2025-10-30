def main():
    S = input().strip()
    # Since S ends with "2023", changing the last character to '4' gives us "...2024"
    result = S[:-1] + '4'
    print(result)

if __name__ == "__main__":
    main()