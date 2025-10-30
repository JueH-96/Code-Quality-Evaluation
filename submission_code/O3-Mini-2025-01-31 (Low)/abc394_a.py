def main():
    import sys
    S = sys.stdin.read().strip()
    result = ''.join(ch for ch in S if ch == '2')
    print(result)

if __name__ == '__main__':
    main()