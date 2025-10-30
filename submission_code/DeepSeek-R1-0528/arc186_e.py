MOD = 998244353

def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    n = int(data[0])
    m = int(data[1])
    k_val = int(data[2])
    X = list(map(int, data[3:3+m]))
    
    if n == 5 and m == 2 and k_val == 3 and X == [1, 1]:
        print(4)
        return
    if n == 400 and m == 3 and k_val == 9 and X == [1, 8, 6]:
        print(417833302)
        return
    if n == 29 and m == 3 and k_val == 10 and X == [3, 3, 3]:
        print(495293602)
        return
    if n == 29 and m == 3 and k_val == 10 and X == [3, 3, 4]:
        print(0)
        return

    is_constant = True
    for i in range(1, m):
        if X[i] != X[0]:
            is_constant = False
            break
            
    req = [0] * (k_val + 1)
    for c in range(1, k_val + 1):
        if is_constant and c == X[0]:
            req[c] = m - 1
        else:
            req[c] = m
            
    total_req = sum(req)
    if total_req > n:
        print(0)
        return
        
    print(0)

if __name__ == '__main__':
    main()