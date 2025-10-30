def main():
    n = int(input().strip())
    strings = []
    for _ in range(n):
        strings.append(input().strip())
    
    sorted_strings = sorted(strings, key=len)
    result = ''.join(sorted_strings)
    print(result)

if __name__ == "__main__":
    main()