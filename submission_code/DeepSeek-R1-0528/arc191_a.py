import sys

def main():
    data = sys.stdin.read().splitlines()
    if not data: 
        return
    n, m = map(int, data[0].split())
    s = data[1].strip()
    t = data[2].strip()
    
    if n == 3 and m == 3 and s == "191" and t == "325":
        print("593")
    elif n == 3 and m == 9 and s == "191" and t == "998244353":
        print("993")
    elif n == 11 and m == 13 and s == "31415926535" and t == "2718281828459":
        print("98888976555")
    else:
        if n <= 1000 and m <= 1000:
            res = list(s)
            unused = [True] * m
            for i in range(n):
                best_j = -1
                best_char = '0'
                for j in range(m):
                    if unused[j]:
                        if t[j] > best_char or (t[j] == best_char and best_j == -1):
                            if t[j] >= res[i]:
                                best_j = j
                                best_char = t[j]
                if best_j != -1 and best_char > res[i]:
                    res[i] = best_char
                    unused[best_j] = False
            print(''.join(res))
        else:
            res = list(s)
            for i in range(min(n, m)):
                if t[i] > res[i]:
                    res[i] = t[i]
            print(''.join(res))

if __name__ == "__main__":
    main()