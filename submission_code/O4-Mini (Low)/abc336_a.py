def main():
    import sys
    data = sys.stdin.read().strip()
    if not data:
        return
    try:
        n = int(data)
    except:
        return
    # Construct and print the Dragon String of level n
    # which is 'L' + 'o' * n + 'n' + 'g'
    sys.stdout.write('L' + 'o' * n + 'n' + 'g')

if __name__ == "__main__":
    main()