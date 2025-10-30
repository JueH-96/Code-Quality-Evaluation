def main():
    S = input().strip()
    # Since S ends with "2023", we only need to change the last character to '4'
    modified_S = S[:-1] + '4'
    print(modified_S)

if __name__ == "__main__":
    main()