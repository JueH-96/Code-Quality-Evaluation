def solve():
    n = int(input())
    s = input()
    a = [int(c) for c in s]
    
    total_sum = 0
    for i in range(n):
        for j in range(i, n):
            if i == j:
                total_sum += a[i]
            else:
                current_nand = a[i]
                for k in range(i + 1, j + 1):
                    if current_nand == 0 and a[k] == 0:
                        current_nand = 1
                    elif current_nand == 0 and a[k] == 1:
                        current_nand = 1
                    elif current_nand == 1 and a[k] == 0:
                        current_nand = 1
                    elif current_nand == 1 and a[k] == 1:
                        current_nand = 0
                total_sum += current_nand
    print(total_sum)

solve()