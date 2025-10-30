def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    if input_data:
        N = int(input_data[0])
        # Generate the alternating string by repeating "10" for N times and then appending an extra "1"
        result = "10" * N + "1"
        print(result)

if __name__ == '__main__':
    main()