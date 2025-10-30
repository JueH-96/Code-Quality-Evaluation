def main():
    s = input().strip()
    parts = s.split('|')
    result = []
    for i in range(1, len(parts) - 1):
        count = len(parts[i])
        result.append(str(count))
    print(" ".join(result))

if __name__ == '__main__':
    main()