def main():
    import sys
    S = sys.stdin.read().strip()
    parts = S.split('.')
    print(parts[-1])

if __name__ == '__main__':
    main()