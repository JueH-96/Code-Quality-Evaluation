def main():
    s = input().strip()
    if s.count('1') == 1 and s.count('2') == 2 and s.count('3') == 3:
        print("Yes")
    else:
        print("No")

if __name__ == '__main__':
    main()