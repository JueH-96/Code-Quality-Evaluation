def main():
    data = input().split()
    c1 = data[1]
    c2 = data[2]
    s = input().strip()
    result = ''.join([char if char == c1 else c2 for char in s])
    print(result)

if __name__ == "__main__":
    main()