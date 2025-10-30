def main():
    import sys
    S = sys.stdin.read().strip()
    # Replace the last character (which is '3') with '4'
    modified = S[:-1] + '4'
    sys.stdout.write(modified)

if __name__ == '__main__':
    main()