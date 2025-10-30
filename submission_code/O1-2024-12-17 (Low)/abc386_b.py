def main():
    import sys
    S = sys.stdin.readline().strip()
    
    i = 0
    presses = 0
    length = len(S)
    
    while i < length:
        # If current char is '0' and next is also '0', use "00" button.
        if S[i] == '0' and i + 1 < length and S[i + 1] == '0':
            presses += 1
            i += 2
        else:
            presses += 1
            i += 1
    
    print(presses)

# Don't forget to call main!
if __name__ == "__main__":
    main()