def main():
    import sys
    data = sys.stdin.read().split()
    N = int(data[0])
    S = data[1]
    T = data[2]
    
    def similar(a, b):
        if a == b:
            return True
        if (a == '1' and b == 'l') or (a == 'l' and b == '1'):
            return True
        if (a == '0' and b == 'o') or (a == 'o' and b == '0'):
            return True
        return False
    
    for ch1, ch2 in zip(S, T):
        if not similar(ch1, ch2):
            print("No")
            return
    print("Yes")

if __name__ == '__main__':
    main()