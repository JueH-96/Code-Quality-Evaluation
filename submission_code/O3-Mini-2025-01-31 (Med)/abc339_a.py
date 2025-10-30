def main():
    import sys
    input_data = sys.stdin.read().strip()
    # Since the string doesn't end with '.', the last substring of split is guaranteed non-empty.
    # Split the string on the '.' character.
    parts = input_data.split('.')
    # The answer is the last element of the split list.
    print(parts[-1])

if __name__ == '__main__':
    main()