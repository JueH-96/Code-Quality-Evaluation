import sys

def main():
    input = sys.stdin.read().split()
    idx = 0
    t = int(input[idx])
    idx += 1
    for _ in range(t):
        S = input[idx]
        idx += 1
        X = input[idx]
        idx += 1
        Y = input[idx]
        idx += 1
        
        a = X.count('0')
        b = X.count('1')
        c = Y.count('0')
        d = Y.count('1')
        A = a - c
        B = d - b
        len_S = len(S)
        
        if B == 0:
            if A != 0:
                print("No")
                continue
            else:
                if X == Y:
                    print("Yes")
                else:
                    print("No")
                continue
        else:
            if (A * len_S) % B != 0:
                print("No")
                continue
            t_len = (A * len_S) // B
            if t_len < 0:
                print("No")
                continue
        
        x_tokens = list(X)
        y_tokens = list(Y)
        i = 0
        j = 0
        s_ptr = 0
        t_ptr = 0
        possible = True
        T = ""  # Dummy, we won't construct it
        
        while i < len(x_tokens) and j < len(y_tokens):
            x_bit = x_tokens[i]
            y_bit = y_tokens[j]
            
            if x_bit == '0' and y_bit == '0':
                i += 1
                j += 1
                s_ptr += len_S
            elif x_bit == '0' and y_bit == '1':
                remaining_S = len_S - s_ptr
                if remaining_S <= 0:
                    possible = False
                    break
                t_ptr += remaining_S
                if t_ptr > t_len:
                    possible = False
                    break
                j += 1
                s_ptr = 0
            elif x_bit == '1' and y_bit == '0':
                remaining_T = t_len - t_ptr
                if remaining_T <= 0:
                    possible = False
                    break
                s_ptr += remaining_T
                if s_ptr > len_S:
                    possible = False
                    break
                i += 1
                t_ptr = 0
            else:
                i += 1
                j += 1
                t_ptr += t_len
        
        if possible and i == len(x_tokens) and j == len(y_tokens):
            print("Yes")
        else:
            print("No")

if __name__ == '__main__':
    main()