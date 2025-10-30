import math

def main():
    R = int(input().strip())
    total = 0
    for i in range(0, R):
        A = 4 * R * R - 4 * i * i - 4 * i - 1
        root = math.isqrt(A)
        j_max = (root - 1) // 2
        if j_max < 0:
            count_i = 0
        else:
            count_i = 2 * j_max + 1
        
        if i == 0:
            total += count_i
        else:
            total += 2 * count_i
            
    print(total)

if __name__ == "__main__":
    main()