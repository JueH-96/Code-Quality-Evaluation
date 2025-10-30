def main():
    import sys
    S = sys.stdin.readline().strip()
    if not S:
        return
    # Take the first character and append "UPC"
    result = S[0] + "UPC"
    print(result)

if __name__ == "__main__":
    main()