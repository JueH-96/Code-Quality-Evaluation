def main():
    S = input().strip()
    # Check every even-positioned character in S.
    # In Python, S[1] is the 2nd character, S[3] is the 4th, and so on.
    for i in range(1, 16, 2):  # indices 1, 3, 5, ... 15 correspond to positions 2, 4, ... 16
        if S[i] != '0':
            print("No")
            return
    print("Yes")
    
if __name__ == '__main__':
    main()