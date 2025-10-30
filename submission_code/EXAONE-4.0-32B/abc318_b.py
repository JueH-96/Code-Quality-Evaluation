def main():
    import sys
    data = sys.stdin.read().split()
    n = int(data[0])
    rectangles = []
    xs_set = set()
    ys_set = set()
    index = 1
    for _ in range(n):
        A = int(data[index])
        B = int(data[index+1])
        C = int(data[index+2])
        D = int(data[index+3])
        index += 4
        rectangles.append((A, B, C, D))
        xs_set.add(A)
        xs_set.add(B)
        ys_set.add(C)
        ys_set.add(D)
    
    xs = sorted(xs_set)
    ys = sorted(ys_set)
    
    total_area = 0
    for i in range(len(xs) - 1):
        x_low = xs[i]
        x_high = xs[i+1]
        for j in range(len(ys) - 1):
            y_low = ys[j]
            y_high = ys[j+1]
            covered = False
            for rect in rectangles:
                A, B, C, D = rect
                if A <= x_low and x_high <= B and C <= y_low and y_high <= D:
                    covered = True
                    break
            if covered:
                total_area += (x_high - x_low) * (y_high - y_low)
    print(total_area)

if __name__ == "__main__":
    main()