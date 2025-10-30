def main():
    import sys
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    s = input_data[0]
    vowels = set("aeiou")
    result = ''.join([char for char in s if char not in vowels])
    sys.stdout.write(result)

if __name__ == '__main__':
    main()