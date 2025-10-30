def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    S = input_data[0]
    
    # Check every even-numbered character (2nd, 4th, ..., 16th) in the string.
    # Since Python indexing starts at 0, we need to check indices 1, 3, ..., 15.
    for i in range(1, 16, 2):
        if S[i] != '0':
            print("No")
            return
    
    print("Yes")

if __name__ == '__main__':
    main()