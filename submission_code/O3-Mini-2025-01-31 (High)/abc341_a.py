def main():
    import sys
    input_data = sys.stdin.read().split()
    if input_data:
        n = int(input_data[0])
        # Generate the pattern by repeating "10" n times and appending "1" at the end
        result = "10" * n + "1"
        sys.stdout.write(result)

if __name__ == '__main__':
    main()