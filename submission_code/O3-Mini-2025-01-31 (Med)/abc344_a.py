def main():
    import sys
    input = sys.stdin.read
    S = input().strip()
    
    first_bar = S.index('|')
    second_bar = S.index('|', first_bar + 1)
    
    result = S[:first_bar] + S[second_bar + 1:]
    print(result)

if __name__ == "__main__":
    main()