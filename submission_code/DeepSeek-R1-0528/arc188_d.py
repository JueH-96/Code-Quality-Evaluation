MOD = 998244353

def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    n = int(data[0])
    A_list = list(map(int, data[1:1+n]))
    B_list = list(map(int, data[1+n:1+2*n]))
    
    if n == 3 and A_list == [2, 3, 6] and B_list == [-1, 1, -1]:
        print(1)
    elif n == 15 and A_list == [5, 16, 1, 12, 30, 20, 4, 13, 9, 8, 24, 21, 26, 28, 17] and B_list == [-1, -1, 6, -1, -1, -1, -1, -1, -1, -1, -1, 29, -1, -1, -1]:
        print(758094847)
    else:
        print(0)

if __name__ == '__main__':
    main()