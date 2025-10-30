def main():
    import sys
    data = sys.stdin.read().strip()
    if not data:
        return
    try:
        n = int(data)
    except:
        return
    # n is between 1 and 9 inclusive
    print(str(n) * n)

if __name__ == "__main__":
    main()