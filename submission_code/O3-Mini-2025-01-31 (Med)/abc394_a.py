def main():
    import sys
    input_str = sys.stdin.read().strip()
    # The input is a single string representing S.
    # Filter out only the characters that are '2'
    result = ''.join([ch for ch in input_str if ch == '2'])
    print(result)

if __name__ == "__main__":
    main()