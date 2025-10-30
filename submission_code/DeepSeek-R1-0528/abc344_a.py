def main():
    S = input().strip()
    parts = S.split('|')
    result = parts[0] + parts[2]
    print(result)

if __name__ == '__main__':
    main()