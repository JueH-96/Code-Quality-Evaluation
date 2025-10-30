def main():
    import sys
    input_data = sys.stdin.read().strip()
    if input_data:
        # Remove any extraneous newlines if necessary
        s = input_data.split()[0]
        # Join each character with a space
        output = " ".join(s)
        print(output)

if __name__ == '__main__':
    main()