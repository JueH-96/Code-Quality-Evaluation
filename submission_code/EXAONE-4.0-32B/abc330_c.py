import math

def main():
    D = int(input().strip())
    ans = D
    x = 0
    while True:
        x2 = x * x
        if x2 > D + ans:
            break
            
        rem = D - x2
        if rem < 0:
            candidate = x2 - D
        else:
            y0 = math.isqrt(rem)
            candidate1 = rem - y0 * y0
            candidate2 = (y0 + 1) * (y0 + 1) - rem
            candidate = min(candidate1, candidate2)
        
        if candidate < ans:
            ans = candidate
            if ans == 0:
                break
                
        x += 1
        
    print(ans)

if __name__ == "__main__":
    main()