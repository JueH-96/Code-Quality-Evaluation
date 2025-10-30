def main():
    import sys
    input_data = sys.stdin.read().strip()
    if input_data:
        N = int(input_data)
        result = "L" + "o" * N + "ng"
        sys.stdout.write(result)

if __name__ == '__main__':
    main()