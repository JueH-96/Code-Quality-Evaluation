def main():
    S = input().strip()
    # Keep only the characters '2'
    result = ''.join(ch for ch in S if ch == '2')
    print(result)

if __name__ == "__main__":
    main()