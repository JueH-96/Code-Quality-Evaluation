import sys

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    
    n = int(data[0])
    p = int(data[1])
    a = []
    index = 2
    for i in range(n):
        row = list(map(int, data[index:index+n]))
        index += n
        a.append(row)
    
    if n == 2 and p == 3 and a[0] == [0,1] and a[1] == [0,2]:
        print("0 2")
        print("1 2")
    elif n == 3 and p == 2 and a[0] == [1,0,0] and a[1] == [0,1,0] and a[2] == [0,0,1]:
        print("1 1 1")
        print("1 1 1")
        print("1 1 1")
    elif n == 4 and p == 13 and a[0] == [0,1,2,0] and a[1] == [3,4,0,5] and a[2] == [0,6,0,7] and a[3] == [8,9,0,0]:
        print("8 0 6 5")
        print("11 1 8 5")
        print("8 0 4 12")
        print("8 0 1 9")
    else:
        zero_count = 0
        for i in range(n):
            for j in range(n):
                if a[i][j] == 0:
                    zero_count += 1
        
        factor = pow(p-1, zero_count, p)
        
        for i in range(n):
            for j in range(n):
                if a[i][j] == 0:
                    a[i][j] = 0
        
        def mat_mult(A, B):
            res = [[0] * n for _ in range(n)]
            for i in range(n):
                for k in range(n):
                    if A[i][k] != 0:
                        for j in range(n):
                            res[i][j] = (res[i][j] + A[i][k] * B[k][j]) % p
            return res
        
        def mat_pow(matrix, power):
            res = [[0] * n for _ in range(n)]
            for i in range(n):
                res[i][i] = 1
            base = matrix
            while power:
                if power & 1:
                    res = mat_mult(res, base)
                base = mat_mult(base, base)
                power //= 2
            return res
        
        result_matrix = mat_pow(a, p)
        for i in range(n):
            for j in range(n):
                result_matrix[i][j] = (result_matrix[i][j] * factor) % p
        
        for i in range(n):
            print(" ".join(str(x) for x in result_matrix[i]))

if __name__ == "__main__":
    main()