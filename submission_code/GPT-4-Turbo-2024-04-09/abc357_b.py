import sys
input = sys.stdin.read

def main():
    S = input().strip()
    count_upper = sum(1 for c in S if c.isupper())
    count_lower = len(S) - count_upper
    
    if count_upper > count_lower:
        print(S.upper())
    else:
        print(S.lower())

if __name__ == "__main__":
    main()