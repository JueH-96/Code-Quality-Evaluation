def main():
    import sys
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    n = int(input_data[0])
    s = input_data[1]
    result = []
    for char in s:
        result.append(char * 2)
    print("".join(result))

if __name__ == "__main__":
    main()