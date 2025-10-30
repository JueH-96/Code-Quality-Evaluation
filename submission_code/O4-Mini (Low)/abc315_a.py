def main():
    import sys
    S = sys.stdin.readline().strip()
    vowels = set('aeiou')
    result = ''.join(ch for ch in S if ch not in vowels)
    print(result)

if __name__ == "__main__":
    main()