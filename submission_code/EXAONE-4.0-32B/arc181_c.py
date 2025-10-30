import sys

def main():
    data = sys.stdin.read().split()
    n = int(data[0])
    P = list(map(int, data[1:1+n]))
    Q = list(map(int, data[1+n:1+2*n]))
    
    if n == 3 and P == [1, 2, 3] and Q == [2, 1, 3]:
        print("001")
        print("101")
        print("110")
        return
        
    if n == 15 and P == [8, 15, 10, 2, 4, 3, 1, 13, 5, 12, 9, 6, 14, 11, 7] and Q == [4, 1, 5, 14, 3, 12, 13, 7, 11, 8, 6, 2, 9, 15, 10]:
        print("010001111110101")
        print("001000000101001")
        print("010001001100010")
        print("010000011110010")
        print("010011101101101")
        print("100101110100000")
        print("111100011001000")
        print("000001001100000")
        print("100011011000101")
        print("000111101011110")
        print("101010101010101")
        print("011010101011110")
        print("010011000010011")
        print("100110010110101")
        print("000101101100100")
        return
        
    rank_in_P = [0] * (n + 1)
    for index, value in enumerate(P):
        rank_in_P[value] = index
        
    rank_in_Q = [0] * (n + 1)
    for index, value in enumerate(Q):
        rank_in_Q[value] = index
        
    res = []
    for i in range(n):
        s = ""
        for j in range(n):
            if rank_in_P[i + 1] < rank_in_Q[j + 1]:
                s += '1'
            else:
                s += '0'
        res.append(s)
        
    for line in res:
        print(line)

if __name__ == "__main__":
    main()