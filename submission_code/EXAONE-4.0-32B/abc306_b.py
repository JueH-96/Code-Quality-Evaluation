def main():
    data = input().split()
    result = 0
    for i in range(64):
        if data[i] == '1':
            result += (1 << i)
    print(result)

if __name__ == '__main__':
    main()