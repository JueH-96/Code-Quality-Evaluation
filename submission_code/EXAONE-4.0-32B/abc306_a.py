def main():
    n_line = input().strip()
    s = input().strip()
    result = ''.join(char * 2 for char in s)
    print(result)

if __name__ == "__main__":
    main()