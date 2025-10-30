def main():
    import sys
    input_data = sys.stdin.read().strip()
    s = input_data.strip()
    vowels = set('aeiou')
    result = ''.join([ch for ch in s if ch not in vowels])
    print(result)

if __name__ == "__main__":
    main()