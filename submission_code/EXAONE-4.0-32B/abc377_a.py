def main():
    s = input().strip()
    if sorted(s) == ['A', 'B', 'C']:
        print("Yes")
    else:
        print("No")

if __name__ == '__main__':
    main()