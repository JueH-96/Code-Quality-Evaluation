import math

def main():
    R = int(input().strip())
    total = 0
    for i in range(R):
        T = 4 * R * R - (2*i + 1) ** 2
        y_max = math.isqrt(T)
        if y_max < 1:
            continue
        j_max = (y_max - 1) // 2
        if i == 0:
            total += 1 + 2 * j_max
        else:
            total += 2 + 4 * j_max
    print(total)

if __name__ == '__main__':
    main()