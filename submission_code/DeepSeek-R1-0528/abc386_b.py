def main():
    S = input().strip()
    count = 0
    total_presses = 0
    for char in S:
        if char == '0':
            count += 1
        else:
            total_presses += (count + 1) // 2
            count = 0
            total_presses += 1
    total_presses += (count + 1) // 2
    print(total_presses)

if __name__ == '__main__':
    main()