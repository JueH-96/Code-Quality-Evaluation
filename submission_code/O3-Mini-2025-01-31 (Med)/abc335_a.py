def main():
    import sys
    input_str = sys.stdin.read().strip()
    # Replace last character with '4'
    output_str = input_str[:-1] + '4'
    print(output_str)

if __name__ == '__main__':
    main()