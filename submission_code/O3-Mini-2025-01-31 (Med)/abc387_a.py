def main():
    import sys
    input_data = sys.stdin.read().strip()
    if input_data:
        A, B = map(int, input_data.split())
        result = (A + B) ** 2
        print(result)

if __name__ == '__main__':
    main()