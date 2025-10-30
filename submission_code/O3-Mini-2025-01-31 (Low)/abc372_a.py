def main():
    import sys
    S = sys.stdin.read().strip()
    # Remove all '.' from S
    result = S.replace('.', '')
    print(result)

if __name__ == "__main__":
    main()