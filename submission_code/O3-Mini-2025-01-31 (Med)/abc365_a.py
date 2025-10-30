def main():
    import sys
    input_data = sys.stdin.read().strip()
    if input_data:
        Y = int(input_data)
        if Y % 400 == 0:
            print(366)
        elif Y % 100 == 0:
            print(365)
        elif Y % 4 == 0:
            print(366)
        else:
            print(365)

if __name__ == "__main__":
    main()