def main():
    n = int(input().strip())
    strings = []
    for _ in range(n):
        s = input().strip()
        strings.append(s)
    
    strings_sorted = sorted(strings, key=lambda x: len(x))
    result = ''.join(strings_sorted)
    print(result)

if __name__ == '__main__':
    main()