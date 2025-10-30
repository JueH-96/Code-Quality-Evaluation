def main():
    import sys
    S = sys.stdin.readline().rstrip('
')
    # Find the first and second pipe characters
    first = S.find('|')
    second = S.find('|', first + 1)
    # Remove the segment between them inclusive
    result = S[:first] + S[second + 1:]
    print(result)

if __name__ == "__main__":
    main()