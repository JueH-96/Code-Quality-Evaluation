def main():
    s = input().strip()
    # The string must have exactly one 'A', one 'B', and one 'C'
    if sorted(s) == sorted("ABC"):
        print("Yes")
    else:
        print("No")

if __name__ == '__main__':
    main()