def solve():
    b = int(input())
    for a in range(1, 16):
        power_val = pow(a, a)
        if power_val == b:
            print(a)
            return
        elif power_val > b:
            break
    print("-1")

if __name__ == '__main__':
    solve()