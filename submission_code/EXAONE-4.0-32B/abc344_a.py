def main():
    S = input().strip()
    first_index = S.find('|')
    second_index = S.find('|', first_index + 1)
    result = S[:first_index] + S[second_index + 1:]
    print(result)

if __name__ == '__main__':
    main()