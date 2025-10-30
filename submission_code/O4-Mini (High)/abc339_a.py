def main():
    import sys
    S = sys.stdin.readline().strip()
    # Find the last dot and print everything after it
    last_dot = S.rfind('.')
    print(S[last_dot+1:])

if __name__ == "__main__":
    main()