def solve():
    n = int(input())
    s = input()
    a = []
    for char in s:
        a.append(int(char))
    
    def nand_op(x, y):
        if x == 1 and y == 1:
            return 0
        else:
            return 1
            
    total_sum = 0
    for i in range(n):
        current_f = a[i]
        total_sum += current_f
        for j in range(i + 1, n):
            current_f = nand_op(current_f, a[j])
            total_sum += current_f
            
    print(total_sum)

if __name__ == '__main__':
    solve()