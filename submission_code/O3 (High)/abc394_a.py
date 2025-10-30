def main():
    import sys
    s = sys.stdin.readline().strip()
    # Keep only '2's and output them in original order
    result = ''.join(ch for ch in s if ch == '2')
    print(result)

if __name__ == "__main__":
    main()