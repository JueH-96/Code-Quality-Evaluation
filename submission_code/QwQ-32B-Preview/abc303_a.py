def are_similar(char1, char2):
    if char1 == char2:
        return True
    elif (char1 == '1' and char2 == 'l') or (char1 == 'l' and char2 == '1'):
        return True
    elif (char1 == '0' and char2 == 'o') or (char1 == 'o' and char2 == '0'):
        return True
    else:
        return False

def main():
    import sys
    try:
        N = int(sys.stdin.readline())
        S = sys.stdin.readline().strip()
        T = sys.stdin.readline().strip()
        
        similar = True
        for i in range(N):
            if not are_similar(S[i], T[i]):
                similar = False
                break
        if similar:
            print("Yes")
        else:
            print("No")
    except:
        print("No")

if __name__ == "__main__":
    main()