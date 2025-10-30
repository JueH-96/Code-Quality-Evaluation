def main():
    import sys
    S = sys.stdin.readline().rstrip()
    # Split on the last '.' and print the segment after it
    print(S.rsplit('.', 1)[-1])

if __name__ == "__main__":
    main()