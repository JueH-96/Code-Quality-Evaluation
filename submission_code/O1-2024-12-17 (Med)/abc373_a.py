def main():
    strings = [input().strip() for _ in range(12)]
    count = 0
    for i, s in enumerate(strings, start=1):
        if len(s) == i:
            count += 1
    print(count)

main()