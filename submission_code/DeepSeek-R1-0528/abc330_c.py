import math

def main():
    D = int(input().strip())
    x_max = math.isqrt(D)
    ans = abs((x_max + 1) ** 2 - D)
    
    for x in range(0, x_max + 1):
        rem = D - x * x
        if rem < 0:
            break
        y0 = math.isqrt(rem)
        candidate1 = rem - y0 * y0
        y1 = y0 + 1
        candidate2 = y1 * y1 - rem
        current_min = min(candidate1, candidate2)
        if current_min < ans:
            ans = current_min
        if ans == 0:
            break
    print(ans)

if __name__ == '__main__':
    main()