import math

def main():
    n = int(input())
    s = input().strip()
    sorted_s = sorted(s)
    max_num = 10 ** n - 1
    max_k = math.isqrt(max_num)
    count = 0
    
    for k in range(max_k + 1):
        square = k * k
        square_str = str(square)
        padded = square_str.zfill(n)
        if sorted(padded) == sorted_s:
            count += 1
    
    print(count)

if __name__ == "__main__":
    main()