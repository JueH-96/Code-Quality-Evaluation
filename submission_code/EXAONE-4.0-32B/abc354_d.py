def f(x, y):
    x = abs(x)
    y = abs(y)
    a00 = (x // 2) * (y // 2)
    a11 = ((x + 1) // 2) * ((y + 1) // 2)
    return a00 + a11

def main():
    import sys
    data = sys.stdin.read().split()
    A = int(data[0])
    B = int(data[1])
    C = int(data[2])
    D = int(data[3])
    
    total_area = (C - A) * (D - B)
    F_CD = f(C, D)
    F_AD = f(A, D)
    F_CB = f(C, B)
    F_AB = f(A, B)
    black_area_in_rect = F_CD - F_AD - F_CB + F_AB
    result = total_area - black_area_in_rect
    print(result)

if __name__ == "__main__":
    main()